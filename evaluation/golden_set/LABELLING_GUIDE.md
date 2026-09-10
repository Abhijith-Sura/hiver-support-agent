# Golden Set Labelling Guide

## Overview
This document describes how the golden evaluation set of 200 examples was sampled, labelled, and validated for the SpotifyCares AI Support Agent evaluation.

## Sampling Strategy

### Population
- Source: ~5,000 SpotifyCares conversation pairs from the Twitter Customer Support dataset (Kaggle)
- Each conversation pair consists of a customer inbound message and the corresponding SpotifyCares reply

### Sampling Method
We use a **stratified sample with oversampling of edge cases**:

| Stratum | Count | Method |
|---------|-------|--------|
| Random proportional | 120 | Uniform random from all conversations, proportional to rough intent distribution |
| Rare intent coverage | 40 | Ensure ≥10 examples per intent category, oversampling underrepresented intents |
| Hard examples | 40 | Deliberately selected edge cases: very short messages (<5 words), very long messages (>50 words), multi-turn threads, ambiguous messages that could map to 2+ intents |

**Total: 200 examples**

### Why this sampling?
- **Random proportional** ensures the evaluation reflects real-world intent distribution
- **Rare intent coverage** prevents rare intents (e.g., family_plan) from having too few examples to measure
- **Hard examples** stress-tests the system on realistic failure cases

## Labelling Protocol

### Labels per example
Each example is labelled with:

1. **`gold_intent`** — One of the 10 intent categories:
   - `account_access`, `playback_issue`, `billing`, `premium_features`, `playlist_library`, `device_sync`, `family_plan`, `bug_report`, `feedback_request`, `other`

2. **`gold_escalation`** — Binary: `"auto_handle"` or `"escalate"`
   - Escalate if: requires account-level access, billing action, customer requests human, legal/safety concern, or issue is too complex for automated response

3. **`gold_reply_quality_notes`** — Free text describing what a good reply should contain

4. **`labeller_notes`** — Free text for ambiguities, edge cases, or disagreements

### Labelling Rules
1. **Read the full thread context** before labelling, not just the last message
2. **Intent = primary intent** — if a message contains multiple issues, label the dominant one
3. **When ambiguous**, label as the more specific intent (e.g., "can't play on my phone" → `device_sync` not `playback_issue` if device-specific)
4. **Escalation is conservative** — when in doubt, label as `escalate` (false negatives are worse than false positives)
5. **The `other` intent** is a last resort — use only when no other intent fits at all

### Quality Assurance
- All 200 examples labelled by a single annotator for consistency
- 50 examples re-reviewed after initial pass to check for drift
- Ambiguous cases documented in `labeller_notes`

## Schema
```json
{
  "id": "gs_001",
  "tweet_id": "119239",
  "customer_message": "My Spotify won't let me log in, keeps saying wrong password",
  "thread_context": [],
  "brand_reply": "@user Hey there! Sorry to hear that. Try resetting your password here: https://t.co/... Let us know if that helps! /AB",
  "gold_intent": "account_access",
  "gold_escalation": "escalate",
  "gold_escalation_reason": "Requires account-level password reset verification",
  "gold_reply_quality_notes": "Should acknowledge frustration, provide password reset link, offer follow-up",
  "labeller_notes": "Clear account access issue, standard resolution path"
}
```
