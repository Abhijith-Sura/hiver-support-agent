"""
Intent taxonomy for SpotifyCares support conversations.

Each intent has a description and example customer messages,
used for LLM few-shot prompting.
"""

INTENT_DEFINITIONS: dict[str, dict] = {
    "account_access": {
        "description": "Login issues, password resets, locked/hacked accounts, account recovery, email changes.",
        "examples": [
            "I can't log into my Spotify account, it keeps saying wrong password",
            "my account got hacked someone changed my email and password help",
            "been locked out of my account for 3 days now, tried resetting password but no email comes through",
        ],
    },
    "playback_issue": {
        "description": "Songs not playing, buffering, audio quality problems, playback stopping unexpectedly, shuffle/repeat issues.",
        "examples": [
            "songs keep pausing every 30 seconds even though I have good wifi",
            "the audio quality sounds terrible today, like I'm listening through a tin can",
            "Spotify won't play any songs, just shows a loading circle forever",
        ],
    },
    "billing": {
        "description": "Unexpected charges, refund requests, payment method issues, subscription pricing, cancellation billing, duplicate charges.",
        "examples": [
            "I was charged $9.99 twice this month, I want a refund for the extra charge",
            "cancelled my subscription last week but just got charged again today??",
            "my credit card expired, how do I update my payment method without losing my playlists",
        ],
    },
    "premium_features": {
        "description": "Questions about upgrading to Premium, feature availability, student/duo plans, ad-free experience, download limits.",
        "examples": [
            "how do I upgrade to premium? the button isn't showing in the app",
            "do students still get 50% off premium? how do I verify my student status",
            "why am I still getting ads even though I'm on the free trial of premium",
        ],
    },
    "playlist_library": {
        "description": "Playlist problems, missing/disappeared songs, library sync issues, liked songs not saving, collaborative playlist issues.",
        "examples": [
            "half my playlist just disappeared overnight, had over 500 songs on it",
            "I keep adding songs to my library but they don't show up on my other devices",
            "my collaborative playlist won't let my friend add songs anymore",
        ],
    },
    "device_sync": {
        "description": "Multi-device playback issues, Spotify Connect problems, offline mode, Bluetooth/speaker/car/smart device integration.",
        "examples": [
            "Spotify keeps switching playback to my laptop when I'm listening on my phone",
            "can't get Spotify connect to work with my Sonos speaker anymore",
            "offline songs won't download on my phone, says waiting to download for hours",
        ],
    },
    "family_plan": {
        "description": "Family plan management, adding/removing family members, address verification, family mix issues.",
        "examples": [
            "I'm trying to add my sister to our family plan but it says the invite link is expired",
            "we all live at the same address but Spotify says we don't for the family plan",
            "one of my family members got kicked off the plan for no reason, can you help",
        ],
    },
    "bug_report": {
        "description": "App crashes, UI glitches, error messages, features not working as expected, update problems.",
        "examples": [
            "the app keeps crashing every time I try to open it after the latest update",
            "there's a weird glitch where the progress bar shows the wrong position in the song",
            "getting error code 4 every time I try to search for something",
        ],
    },
    "feedback_request": {
        "description": "Feature suggestions, general feedback, praise, complaints about product direction, wishlists.",
        "examples": [
            "it would be amazing if you could add a sleep timer to the desktop app",
            "please bring back the old UI, the new one is so confusing",
            "just wanted to say thanks for Spotify Wrapped, it was really cool this year!",
        ],
    },
    "other": {
        "description": "Messages that don't fit any other category: general greetings, unclear messages, non-support queries, spam.",
        "examples": [
            "hey",
            "what time does your office close",
            "lol spotify is the best but also the worst sometimes you know",
        ],
    },
}


def get_taxonomy_prompt() -> str:
    """Format the taxonomy into a system prompt string for LLM classification."""
    lines = [
        "You are a customer support intent classifier for SpotifyCares (Spotify's Twitter support).",
        "Classify each customer message into exactly ONE of the following intents:\n",
    ]
    for intent, defn in INTENT_DEFINITIONS.items():
        lines.append(f"**{intent}**: {defn['description']}")

    lines.append("\nRules:")
    lines.append("- Choose the MOST SPECIFIC intent that fits.")
    lines.append("- If a message mentions multiple issues, pick the PRIMARY one.")
    lines.append("- Use 'other' ONLY when no other intent fits at all.")
    lines.append("- Provide a confidence score (0.0 to 1.0) reflecting your certainty.")
    lines.append("- Provide brief reasoning for your classification.")

    return "\n".join(lines)


def get_few_shot_examples() -> str:
    """Format few-shot examples for the LLM prompt."""
    lines = ["\nExamples:\n"]
    for intent, defn in INTENT_DEFINITIONS.items():
        for i, example in enumerate(defn["examples"][:2]):  # 2 examples per intent
            lines.append(f'Customer: "{example}"')
            lines.append(f'Classification: {{"intent": "{intent}", "confidence": 0.95, '
                         f'"reasoning": "Matches {intent} — {defn["description"][:60]}…"}}')
            lines.append("")
    return "\n".join(lines)


def get_intent_labels() -> list[str]:
    """Return ordered list of intent labels."""
    return list(INTENT_DEFINITIONS.keys())
