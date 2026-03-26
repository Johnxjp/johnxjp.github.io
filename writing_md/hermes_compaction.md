## How Compaction Works in Hermes Agent
March 26, 2026 
Hermes Agent Commit: **f83c27e**

[Nous Research](https://nousresearch.com/) recently released [Hermes Agent](https://github.com/NousResearch/hermes-agent/tree/main) — an open-source personal agent similar to OpenClaw. One of the aspects I was most curious about was context management and in particular compaction given that effective context management is arguably the most critical requirement for maximising agent performance in long-running contexts. In this post, I document Hermes’ approach to compaction: how, where and when. 

### **The How**

Compaction compresses the agent’s current context into a smaller number of tokens. This is usually done out of necessity to ensure the input fits into the LLMs context or for performance as [performance has been shown to decrease](https://www.notion.so/Hermes-Agent-32dd52d026fc804d97cddedcb51f0407?pvs=21) with longer contexts. In theory, there are many ways to shrink the context window from naive implementations like deleting everything or retaining the last few messages to more sophisticated pruning. Getting this right is important to ensuring the agent can effectively continue the task without a performance drop or needing to remind it of the entire context. This is precisely why I was curious about how Hermes implements compaction. Thankfully, Hermes is neatly documented and [Nous tells us](https://github.com/NousResearch/hermes-agent/blob/e4033b2baf681946bc36b3c02546866a28c7aae9/agent/context_compressor.py#L546) exactly how they do it in plain English:

```python
Compress conversation messages by summarizing middle turns.

Algorithm:
  1. Prune old tool results (cheap pre-pass, no LLM call)
  2. Protect head messages (system prompt + first exchange)
  3. Find tail boundary by token budget (~20K tokens of recent context)
  4. Summarize middle turns with structured LLM prompt
  5. On re-compression, iteratively update the previous summary

After compression, orphaned tool_call / tool_result pairs are cleaned
up so the API never receives mismatched IDs.
```

It’s worth describing the overall approach before diving in. Essentially, Hermes Agent chunks up the conversation history into a head, torso and tail. The head and tail are left untouched and the middle portion is summarised. This is actually the same approach OpenClaw takes. Now, how does each part work?

1. *Prune old tool results*

This step is pretty ordinary — go through each old tool call and replace the result with placeholder text where ‘old’ is defined as anything in the middle portion of the context window. More precisely, only long tool results are replaced with the placeholder string `[Old tool output cleared to save context space]` . 

On first glance, it wasn’t immediately obvious why pruning tool results was necessary before I read the above placeholder string. Reducing the size of the context to be compacted could positively impact compression performance. You could argue that tool results could be valuable to have for summarisation, but I guess it’s assumed that results are already sufficiently described in the agent’s conversational messages so that the results themselves aren’t actually sufficiently valuable. [Anthropic also seem to do the same arguing that old tool calls aren't valuable.](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

```python
result = [m.copy() for m in messages]
prune_boundary = len(result) - protect_tail_count
for i in range(prune_boundary):
    msg = result[i]
    if msg.get("role") != "tool":
        continue
    content = msg.get("content", "")
    if not content or content == _PRUNED_TOOL_PLACEHOLDER:
        continue
    # Only prune if the content is substantial (>200 chars)
    if len(content) > 200:
        result[i] = {**msg, "content": _PRUNED_TOOL_PLACEHOLDER}
```

  *2. Protect head messages*

[This one](https://github.com/NousResearch/hermes-agent/blob/f83c27e26f22e34b4b6337bb45608caf5a02e9c6/agent/context_compressor.py#L579) is pretty simple. There isn’t really a point in summarising the system prompt as it’s independent of the conversation and I guess first few user messages shape the entire task. The default number of messages in the head is set to 3 but the precise number can vary depending on tool call behaviour. The actual algorithm ensures that the last message in the head is not a tool result and instead the head size increases to ensure the middle region doesn’t start with an orphaned tool call or result. Not much more to say here.

  *3. Protect the tail messages*

The tail is also reserved because it contains the highest signal of what the agent was most recently doing. It might be hazardous to poke holes in this and compress it into a lossy signal. One interesting design choice here is that the size of the tail is defined by the number of tokens instead of number of messages. This allows the tail to scale with the context and ensures significant summarisation can occur despite model size. Imagine if the model had small context window and the number of messages in the tail consumed a significant portion. Similar to the head, the boundary is also slightly shifted to ensure tool call blocks are grouped.

  *4. Summarise the middle* 

Now we get to the heart of the algorithm. The middle portion of the message history is passed to an LLM which creates a summary based on a structured template. The template asks the LLM to preserve:

- The current goal
- Constraints and user preferences
- Progress towards the goal including tasks complete, in-progress and blocked
- Key decisions that have been made
- Relevant references e.g. files
- Next steps
- Other critical context e.g. config details

The summary can change depending on if there was a previous compaction event that already created a summary. The model used in summarisation is by default the same model as the head agent but another model could also be used. Using the current model prevents problems like mismatched context window sizes or out-of-distribution errors e.g. summarisation model doesn't understand code very well. If a user was to select a model that can't handle the context then the middle portions are simply dropped. As for the risk of out-of-distribution hits, it seems higher in a system like Hermes because it is a general purpose agent that operates across a wide-variety of tasks. In practice, I’m not entirely sure how worrying ‘out-of-distribution’ really is although I’m certain summarisation quality is impacted by model selection and so like everything in AI, it’s best to rely on empirical evidence by evaluating.

Finally, the output summary is returned and appended to a prefix that signals to the model a compaction event happened.

```python
SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION] Earlier turns in this conversation were compacted "
    "to save context space. The summary below describes work that was "
    "already completed, and the current session state may still reflect "
    "that work (for example, files may already be changed). Use the summary "
    "and the current state to continue from where things left off, and "
    "avoid repeating work:"
)
```

  *5. Assemble the compressed message*

Stitching all the pieces together requires a small amount of work. One of the aspects of this stage is to ensure that the messages alternate between ‘user’ and ‘assistant’ because this is what LLMs have been trained to expect and this is what’s required. The role of the summary message therefore is chosen based on whichever would preserve this pattern.

```python
last_head_role = messages[compress_start - 1].get("role", "user") if compress_start > 0 else "user"
first_tail_role = messages[compress_end].get("role", "user") if compress_end < n_messages else "user"
# Pick a role that avoids consecutive same-role with both neighbors.
# Priority: avoid colliding with head (already committed), then tail.
if last_head_role in ("assistant", "tool"):
    summary_role = "user"
else:
    summary_role = "assistant"
# If the chosen role collides with the tail AND flipping wouldn't
# collide with the head, flip it.
if summary_role == first_tail_role:
    flipped = "assistant" if summary_role == "user" else "user"
    if flipped != last_head_role:
        summary_role = flipped
    else:
        # Both roles would create consecutive same-role messages
        # (e.g. head=assistant, tail=user — neither role works).
        # Merge the summary into the first tail message instead
        # of inserting a standalone message that breaks alternation.
        _merge_summary_into_tail = True
if not _merge_summary_into_tail:
    compressed.append({"role": summary_role, "content": summary})
```

There’s also a final check to ensure there are no orphaned tool calls or results as provider APIs also typically reject this. 

That covers the core compression algorithm. But compaction doesn't happen in isolation — there's meaningful work before and after it runs.

#### **Pre and Post Compaction Processing**

**Pre-processing**

The above algorithm is compaction in isolation but depending on where the algorithm is invoked there is some pre- and post-processing.  Compaction can be triggered [manually](https://github.com/NousResearch/hermes-agent/blob/36af1f3baf3f2b089ca3bd5c3b9405bdaf9689d6/cli.py#L4334) with the slash command `/compress` or by the system in the agent loop. The manual trigger just calls the agent loop method `_compress_context` so we’ll look at that. The method is [here](https://github.com/NousResearch/hermes-agent/blob/36af1f3baf3f2b089ca3bd5c3b9405bdaf9689d6/run_agent.py#L4819).

Before compaction, the system is prompted to [extract any relevant memories](https://github.com/NousResearch/hermes-agent/blob/36af1f3baf3f2b089ca3bd5c3b9405bdaf9689d6/run_agent.py#L4656) before they are possibly lost. 

```python
 # Pre-compression memory flush: let the model save memories before they're lost
self.flush_memories(messages, min_turns=0)
```

This method actually sends a background user message to nudge an LLM to save any memories worth remembering

```python
flush_content = (
    "[System: The session is being compressed. "
    "Save anything worth remembering — prioritize user preferences, "
    "corrections, and recurring patterns over task-specific details.]"
)
```

There’s some work to prepare the message for different API providers, but essentially an LLM call is made with a single tool `memory_tool_def` to review the entire conversation history and save any valuable memories. The tool definition is [here](https://github.com/NousResearch/hermes-agent/blob/f83c27e26f22e34b4b6337bb45608caf5a02e9c6/tools/memory_tool.py#L476) and is a all-purpose memory tool which gives the LLM access to a memory store and the ability to add, replace or remove items. Any tool calls are executed and then the conversation history is cleaned up to remove the extra elements injected during the flush event. 

**Post-processing**

Once compaction runs a couple of steps are performed to re-establish the conversation for continuation. 

1. The agent’s pending and in-progress task list is appended to the conversation. It’s interesting that this is included here because the agent has a separate TODO store that it can access and tasks are also written into the compaction summary. I can only imagine this is done to lower the odds the agent goes off-track.

```python
    todo_snapshot = self._todo_store.format_for_injection()
    if todo_snapshot:
        compressed.append({"role": "user", "content": todo_snapshot})
```

1. The [system prompt is rebuilt](https://github.com/NousResearch/hermes-agent/blob/f83c27e26f22e34b4b6337bb45608caf5a02e9c6/run_agent.py#L2255) and added to the top of the conversation history. The system prompt is comprised of the user’s memories which may have changed after the flush event before compaction. The cache is also invalidated so that the new system prompt is forced into use instead.

```python
    self._invalidate_system_prompt()
    new_system_prompt = self._build_system_prompt(system_message)
    self._cached_system_prompt = new_system_prompt
```

1. Session records are updated to reflect a compaction event has occurred and counters are reset.

### The Where and When

Compaction runs either inside the agent loop or when [manually triggered](https://github.com/NousResearch/hermes-agent/blob/36af1f3baf3f2b089ca3bd5c3b9405bdaf9689d6/cli.py#L4334) as a slash command `/compress`. Inside the agent loop, compression can occur in two places (in [run_agent.py](https://github.com/NousResearch/hermes-agent/blob/main/run_agent.py)):

1. [Proactive] [Before a new user request is handled (pre-flight)](https://github.com/NousResearch/hermes-agent/blob/87e2626cf6d490f03f48bf44d6d8c324bed56153/run_agent.py#L5555). 
2. [Reactive] [During agent execution](https://github.com/NousResearch/hermes-agent/blob/87e2626cf6d490f03f48bf44d6d8c324bed56153/run_agent.py#L6257) when context grows too large. This is triggered after received an API error e.g. receiving a 413 error code.

The pre-flight compaction is an interesting edge case. This tries to handle the case where the conversation history already exceeds a threshold number of tokens that should trigger compaction and is checked after a new user message arrives. I wasn’t quite sure why this was needed but then understood that a user can manually change the model partway through the conversation and in fact choose a model that has a smaller window compromising the context. The threshold is set by default to be 50% of the current model’s context window size. They also handle the case where multiple compactions might be necessary to reduce the current token size to fit a very small model.

```python
if (
    self.compression_enabled
    and len(messages) > self.context_compressor.protect_first_n
                        + self.context_compressor.protect_last_n + 1
):
    _sys_tok_est = estimate_tokens_rough(active_system_prompt or "")
    _msg_tok_est = estimate_messages_tokens_rough(messages)
    _preflight_tokens = _sys_tok_est + _msg_tok_est

    if _preflight_tokens >= self.context_compressor.threshold_tokens:
        # May need multiple passes for very large sessions with small
        # context windows (each pass summarises the middle N turns).
        for _pass in range(3):
            _orig_len = len(messages)
            messages, active_system_prompt = self._compress_context(
                messages, system_message, approx_tokens=_preflight_tokens,
                task_id=effective_task_id,
            )
            if len(messages) >= _orig_len:
                break  # Cannot compress further
            # Re-estimate after compression
            _sys_tok_est = estimate_tokens_rough(active_system_prompt or "")
            _msg_tok_est = estimate_messages_tokens_rough(messages)
            _preflight_tokens = _sys_tok_est + _msg_tok_est
            if _preflight_tokens < self.context_compressor.threshold_tokens:
                break  # Under threshold
```

Finally, there’s an upper limit to the number of times compaction can happen per turn which is by default 3 and after which the result is incomplete.

And there you have it. There are likely many alternatives to compaction from approaches like rolling window strategies, more selective exfiltration methods and so forth. This approach favours simplicity and flexibility, which seems like a reasonable approach to take when building such a general purpose agent. I would love to understand what other methods might have been tested and the performance of those relative to thsi approach. If you have any thoughts on compaction or context engineering more generally, please share with me on X at https://x.com/johnlingi.