"""
=============================================================
  STEP 1: DATA GENERATION
  File: backend/data/generate_data.py
=============================================================

PURPOSE:
    Since free-tier projects can't scrape live Twitter/X data,
    we generate a realistic synthetic dataset that mimics
    real social media posts (short, informal, emoji-heavy text).

CONCEPTS COVERED:
    - Dataset creation and structure
    - Class balance (equal samples per sentiment)
    - CSV storage (industry standard for tabular ML data)

HOW IT WORKS:
    1. We define example sentences for each sentiment class
       (positive, negative, neutral)
    2. We randomly combine them with variations (emojis, slang, etc.)
    3. We save everything to a CSV file for training

RUN THIS FIRST before training the model.
"""

import csv          # Built-in: for writing CSV files
import random       # Built-in: for shuffling and random choices
import os           # Built-in: for file path handling


# ─────────────────────────────────────────────
#   SEED for reproducibility
#   (Same seed = same random output every time)
# ─────────────────────────────────────────────
random.seed(42)


# ─────────────────────────────────────────────
#   RAW TRAINING SENTENCES
#   These are hand-crafted examples that mimic
#   real social media language patterns.
# ─────────────────────────────────────────────

POSITIVE_POSTS = [
    "I absolutely love this! Best day ever 😍",
    "This made my whole week, totally amazing 🎉",
    "So happy with how things turned out today!",
    "Can't believe how great this is, highly recommend 👌",
    "Feeling blessed and grateful for everything 🙏",
    "Just had the best coffee of my life ☕ totally worth it",
    "Life is beautiful when you appreciate the little things 🌸",
    "This product changed my life for real, no joke!",
    "Incredible experience, would do it again in a heartbeat",
    "Today was absolutely perfect, nothing could be better",
    "Such a good vibe today, loving every moment 🔥",
    "That new movie was fire, 10/10 no notes 🎬",
    "My team won and I'm over the moon right now 🏆",
    "Just got promoted! Dreams do come true 🎊",
    "The weather is gorgeous, perfect day to be alive 🌞",
    "New album just dropped and it's everything I needed 🎵",
    "So proud of myself for finishing this project!",
    "This restaurant is genuinely amazing, A+ would return",
    "Happiness is contagious and I'm spreading it today 😄",
    "Best decision I ever made, totally no regrets ✅",
    "Finally got my dream job, so excited to start!",
    "This trip was worth every penny, absolutely stunning 🏝️",
    "Love my friends so much, greatest support system 💕",
    "Finished my workout feeling like a champion 💪",
    "Things are finally falling into place, grateful AF",
    "That sunset was absolutely breathtaking 🌅",
    "New personal record! Hard work really pays off",
    "Woke up feeling refreshed and ready to crush it",
    "The kindness of strangers still surprises me every day",
    "Positive energy only, life is too short for anything else",
]

NEGATIVE_POSTS = [
    "I hate how this turned out, completely disappointed 😠",
    "Worst experience of my life, would not recommend at all",
    "This is absolutely terrible and I want my money back",
    "Can't believe I wasted my time on this garbage 🗑️",
    "So frustrated right now, nothing is working out",
    "This service is trash, the worst I have ever seen",
    "Feeling so down and hopeless today, everything sucks",
    "Lost my phone and wallet on the same day, kill me",
    "This product is a total scam, stay far away",
    "I'm beyond done with this situation, completely over it",
    "Today was an absolute disaster from start to finish",
    "Why does everything always go wrong for me specifically",
    "This is a disgrace, I am never coming back here again",
    "Genuinely the worst customer service I've experienced",
    "So angry I could scream, this is unacceptable behavior",
    "Rain ruined everything I had planned, totally miserable ☔",
    "Failed my exam after studying for weeks, heartbroken 💔",
    "My flight got cancelled and I'm stuck here, awful",
    "The movie was absolutely dreadful, wasted 2 hours of my life",
    "I'm exhausted and nothing seems to be going right lately",
    "Traffic was a nightmare and made me miss my meeting",
    "The food was cold and the waiter was extremely rude",
    "I can't stop crying and I don't even know why anymore",
    "Burned out and completely over everything at this point",
    "This is the third time this week something has broken",
    "So let down by someone I trusted completely, betrayal hurts",
    "My team lost again, at this point it's embarrassing",
    "Worst birthday ever, nothing went the way it was planned",
    "Overcharged again by this company, absolutely ridiculous",
    "I'm done pretending everything is fine when it clearly isn't",
]

NEUTRAL_POSTS = [
    "Just woke up and making some coffee, another Monday",
    "The weather today is kind of okay, nothing special",
    "Heading to work now, same routine as always",
    "Ordered pizza for dinner tonight, nothing crazy",
    "The meeting ran a bit long but it ended eventually",
    "Read an article about the economy, interesting stuff",
    "Watched some TV and went to bed at a normal time",
    "The package arrived, it is what it is I guess",
    "Finished the report, submitted it, moving on now",
    "Had lunch at that new place downtown, it was fine",
    "Traffic was average today, nothing to write home about",
    "Getting ready for the weekend, no big plans really",
    "The update installed correctly, everything seems normal",
    "Went for a walk around the neighborhood this afternoon",
    "Tried a new recipe, it turned out okay I suppose",
    "There is a sale at the mall, might check it out later",
    "The library was quiet today, got some work done",
    "Watched the news for a bit, lots going on in the world",
    "Called my parents this evening, same as usual",
    "Did some laundry and cleaned up a bit, productive enough",
    "The app updated and the interface looks slightly different",
    "Finished the book I was reading, it was decent",
    "Went grocery shopping, nothing unusual on the list",
    "The commute was about what I expected, nothing more",
    "Checked my email, replied to a few things, moved on",
    "Temperature is around average for this time of year",
    "The presentation is scheduled for next Thursday morning",
    "Just got back from running some errands around town",
    "The new software version seems to work about the same",
    "Had a regular day at the office, nothing stood out",
]


def generate_dataset(output_path: str, samples_per_class: int = 200) -> None:
    """
    Generate a balanced sentiment dataset and save it as CSV.

    Args:
        output_path     : Where to save the CSV file
        samples_per_class: How many rows per sentiment class
                           (equal counts = balanced dataset)

    A BALANCED DATASET is important in ML because:
        - If you have 900 "positive" and 50 "negative" examples,
          the model learns to just predict "positive" all the time.
        - Equal class counts force the model to actually learn
          the differences between classes.
    """

    print("=" * 55)
    print("  GENERATING SYNTHETIC SOCIAL MEDIA DATASET")
    print("=" * 55)

    # ── Build the full dataset by sampling from our template lists ──
    dataset = []

    # Helper: sample with replacement so we can exceed base list size
    def oversample(base_list: list, n: int) -> list:
        """
        If n > len(base_list), we wrap around (sample with replacement).
        This is a simple form of data augmentation.
        """
        return [random.choice(base_list) for _ in range(n)]

    # Generate rows for each class
    for label, source in [
        ("positive", POSITIVE_POSTS),
        ("negative", NEGATIVE_POSTS),
        ("neutral",  NEUTRAL_POSTS),
    ]:
        posts = oversample(source, samples_per_class)
        for text in posts:
            dataset.append({"text": text, "sentiment": label})
        print(f"  ✔ Generated {samples_per_class} '{label}' samples")

    # ── Shuffle dataset so classes aren't in order ──
    # Without shuffling, training might see all positives, then all negatives,
    # which can confuse gradient-based optimizers.
    random.shuffle(dataset)

    # ── Ensure output directory exists ──
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ── Write to CSV ──
    fieldnames = ["text", "sentiment"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)

    total = len(dataset)
    print(f"\n  ✅ Dataset saved → {output_path}")
    print(f"     Total rows : {total}")
    print(f"     Per class  : {samples_per_class}")
    print("=" * 55)


# ─────────────────────────────────────────────
#   ENTRY POINT
#   This block runs only when you execute this
#   file directly: `python generate_data.py`
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # Resolve path relative to this file's location
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "social_media_data.csv")

    generate_dataset(output_path=output_path, samples_per_class=200)