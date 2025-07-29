import streamlit as st
from streamlit_lottie import st_lottie
import re
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from nrclex import NRCLex
import numpy as np
import ijson
from sklearn.metrics.pairwise import cosine_similarity
import random
from io import StringIO
import syllables
import plotly.express as px
import plotly.graph_objects as go
from textblob import Word
import nltk
import logging
from sentence_transformers import SentenceTransformer
from nltk.corpus import wordnet
import spacy
import colorsys
import pronouncing
from itertools import combinations
import uuid
from empath import Empath
from transformers import pipeline
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import time

st.set_page_config(page_title="Mumbai Poets Society Poetry Analyzer", page_icon="📜", layout="wide")

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("poetry_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)  # Added stopwords
except Exception as e:
    logger.error(f"Error downloading NLTK data: {e}")

def load_lottie(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Cache models with enhanced error handling
@st.cache_resource
def load_embedder():
    try:
        model = SentenceTransformer('all-distilroberta-v1', device='cpu')
        logger.info("Sentence transformer loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading sentence transformer: {e}")
        return None

@st.cache_resource
def load_spacy():
    try:
        nlp = spacy.load('en_core_web_lg')
        logger.info("Spacy model loaded successfully")
        return nlp
    except Exception as e:
        logger.error(f"Error loading spacy model: {e}")
        return None

@st.cache_resource
def load_emotion_classifier():
    try:
        classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
        logger.info("Emotion classifier loaded successfully")
        return classifier
    except Exception as e:
        logger.error(f"Error loading emotion classifier: {e}")
        return None

embedder = load_embedder()
nlp = load_spacy()
emotion_classifier = load_emotion_classifier()
empath = Empath()

# Add this after your imports
possible_paths = [
    "songs.json",  # Current directory
    os.path.join(os.path.dirname(__file__), "songs.json"),  # Script directory
    os.path.join(os.path.expanduser("~"), "songs.json"),  # User home directory
    os.path.join(os.path.expanduser("~"), "Downloads", "songs.json")  # Downloads folder
]

# Comprehensive constants
QUOTES = [
    "Poetry is the lifeblood of rebellion, revelation, and love. – Nazim Hikmet",
    "Poetry heals the wounds inflicted by reason. – Novalis",
    "A poem is a painting that is not seen. – Emily Dickinson",
    "Poetry is the shadow cast by our imagination. – Lawrence Ferlinghetti",
    "Words are the only things that last forever. – William Hazlitt",
    "Poetry is the music of the soul. – Voltaire",
    "A poet's work is to name the unnameable. – Salman Rushdie",
    "Poetry is when an emotion has found its thought and the thought has found words. – Robert Frost",
    "To be a poet is a condition, not a profession. – Robert Graves",
    "Poetry is the journal of a sea animal living on land, wanting to fly in the air. – Carl Sandburg"
]
EMOTION_KEYWORDS = {
    "happy": {
        "keywords": ["happy", "joy", "delight", "smile", "cheer", "bliss", "glee", "ecstasy", "radiant", "jubilant", 
                    "laugh", "celebrate", "merry", "elation", "euphoria", "content", "pleasure", "felicity", 
                    "exhilaration", "buoyant", "cheerful", "overjoyed", "thrilled", "gleeful", "jovial", "sunny",
                    "upbeat", "festive", "mirth", "jolly", "exuberant", "vivacious", "sprightly", "chipper"],
        "color": "#FFD700",
        "emoji": "😊"
    },
    "sadness": {
        "keywords": ["sadness", "tears", "sorrow", "grief", "woe", "melancholy", "despair", "heartache", "lament", 
                    "mourn", "weep", "anguish", "forlorn", "desolate", "bereft", "wistful", "regret", "gloom", 
                    "dejection", "suffering", "heartbreak", "misery", "woeful", "doleful", "somber", "bleak",
                    "dismal", "lugubrious", "funereal", "morose", "pensive", "heavy-hearted", "downcast"],
        "color": "#4682B4",
        "emoji": "😢"
    },
    "anger": {
        "keywords": ["anger", "rage", "fury", "ire", "outrage", "wrath", "indignation", "resentment", "frustration", 
                     "hostile", "fuming", "seething", "irate", "enraged", "infuriated", "incensed", "aggravation", 
                     "vexation", "bitterness", "animosity", "choler", "spite", "malice", "scorn", "temper",
                     "outburst", "tantrum", "provocation", "irritation", "exasperation", "displeasure", "gall"],
        "color": "#FF4500",
        "emoji": "😠"
    },
    "calm": {
        "keywords": ["calm", "peace", "serene", "tranquil", "soothe", "stillness", "relax", "quietude", "harmony", 
                     "placid", "composed", "untroubled", "restful", "peaceful", "equanimity", "repose", "calmness", 
                     "gentle", "mellow", "soothing", "zen", "balanced", "steady", "poised", "unruffled",
                     "collected", "cool-headed", "unperturbed", "sedate", "level-headed", "unflappable"],
        "color": "#6B7280",
        "emoji": "😌"
    },
    "fear": {
        "keywords": ["fear", "terror", "dread", "panic", "fright", "anxiety", "apprehension", "unease", "trepidation", 
                     "horror", "phobia", "shudder", "alarm", "nervous", "afraid", "scared", "petrified", "timid", 
                     "worry", "disquiet", "foreboding", "panic-stricken", "terrified", "unnerved", "quaking",
                     "jittery", "jumpy", "skittish", "fearful", "timorous", "quivering", "shaky"],
        "color": "#800080",
        "emoji": "😨"
    },
    "surprise": {
        "keywords": ["surprise", "shock", "amaze", "astonish", "wonder", "startle", "bewilder", "stun", "astound", 
                     "flabbergast", "dumbfound", "marvel", "awe", "disbelief", "gasp", "stagger", "shockaLad", 
                     "unbelievable", "jaw-dropping", "unexpected", "revelation", "eye-opener", "bolt", "bombshell",
                     "whammy", "curveball", "jolt", "revelation", "miracle", "phenomenon", "spectacle"],
        "color": "#FF69B4",
        "emoji": "😲"
    },
    "hopeful": {
        "keywords": ["hope", "dream", "aspire", "optimism", "inspiration", "uplift", "confidence", "yearning", "faith", 
                     "expectation", "promise", "ambition", "encouragement", "vision", "anticipation", "belief", 
                     "desire", "longing", "assurance", "prospect", "renewal", "bright", "buoyed", "heartened",
                     "auspicious", "propitious", "rosy", "upbeat", "sanguine", "forward-looking", "promising"],
        "color": "#FFA07A",
        "emoji": "✨"
    },
    "disgust": {
        "keywords": ["disgust", "repulsion", "revulsion", "nausea", "aversion", "loathing", "abhorrence", "scorn", 
                     "contempt", "repugnance", "detest", "sickening", "offensive", "vile", "revolting", "gross", 
                     "distaste", "hateful", "abominable", "repellent", "odious", "foul", "ghastly", "nauseating",
                     "repulsive", "obnoxious", "abomination", "horrid", "gruesome", "putrid", "rank"],
        "color": "#6B8E23",
        "emoji": "🤢"
    }
}

NRCLEX_TO_EMOTION = {
    "anticipation": "hopeful",
    "trust": "hopeful",
    "joy": "happy",
    "fear": "fear",
    "anger": "anger",
    "sadness": "sadness",
    "disgust": "disgust",
    "surprise": "surprise",
    "positive": "happy",
    "negative": "sadness"
}

THEME_KEYWORDS = {
    "nature": ["tree", "forest", "river", "mountain", "sky", "wind", "sea", "flower", "sun", "moon", "ocean", "cloud", 
               "earth", "breeze", "valley", "meadow", "dawn", "sunset", "horizon", "landscape", "wilderness", "flora", 
               "fauna", "ridge", "canopy", "tide", "bloom", "verdant", "twilight", "aurora", "petal", "thunder",
               "lightning", "drizzle", "monsoon", "meadow", "glade", "thicket", "grove", "copse", "brook", "stream"],
    "love": ["love", "heart", "kiss", "embrace", "darling", "beloved", "romance", "passion", "adore", "cherish", 
             "devotion", "lover", "affection", "desire", "intimacy", "tenderness", "soulmate", "yearning", "flame", 
             "vow", "ardor", "infatuation", "courtship", "sweetheart", "enamored", "fondness", "attachment", "longing",
             "adoration", "endearment", "sentiment", "amour", "pining", "rapture", "belonging", "togetherness"],
    "loss": ["loss", "grief", "death", "mourn", "farewell", "gone", "depart", "sorrow", "lament", "absence", 
             "emptiness", "bereavement", "void", "goodbye", "memorial", "elegy", "remembrance", "longing", "solitude", 
             "heartbreak", "wistful", "epitaph", "ashes", "fading", "severance", "parting", "separation", "divorce",
             "termination", "conclusion", "cessation", "demise", "extinction", "oblivion", "nonexistence"],
    "hope": ["hope", "dream", "future", "light", "promise", "rise", "aspire", "vision", "renewal", "faith", "optimism", 
             "dawn", "rebirth", "recovery", "revival", "resilience", "inspiration", "uplift", "beacon", "prospect", 
             "ambition", "encouragement", "redemption", "spark", "fortitude", "expectation", "potential", "possibility",
             "opportunity", "anticipation", "aspiration", "faith", "confidence", "assurance", "certainty"],
    "time": ["time", "moment", "hour", "day", "eternity", "past", "future", "memory", "fleeting", "age", "forever", 
             "clock", "season", "temporal", "ephemeral", "yesterday", "tomorrow", "decade", "era", "chronicle", 
             "instant", "perpetual", "nostalgia", "continuum", "cycle", "duration", "interval", "phase", "epoch",
             "generation", "lifetime", "millennium", "century", "aeon", "infinity", "permanence"],
    "struggle": ["struggle", "fight", "battle", "pain", "suffer", "hardship", "trial", "conflict", "anguish", "endure", 
                 "resist", "defiance", "overcome", "persevere", "adversity", "tribulation", "resilience", "combat", 
                 "obstacle", "agony", "clash", "toil", "burden", "defy", "tenacity", "opposition", "challenge",
                 "ordeal", "setback", "misfortune", "difficulty", "hardship", "travail", "exertion", "labor"],
    "identity": ["identity", "self", "soul", "being", "essence", "individual", "personhood", "heritage", "roots", 
                 "belonging", "authenticity", "character", "persona", "selfhood", "origin", "legacy", "culture", 
                 "kinship", "pride", "self-discovery", "introspection", "reflection", "ego", "consciousness",
                 "subjectivity", "uniqueness", "individuality", "temperament", "disposition", "personality", "nature"],
    "freedom": ["freedom", "liberty", "emancipation", "release", "escape", "soar", "unbound", "free", "autonomy", 
                "deliverance", "liberation", "independence", "unshackled", "sovereignty", "unrestrained", "flight", 
                "openness", "exodus", "unfettered", "self-determination", "agency", "choice", "will", "volition",
                "autonomy", "self-rule", "self-governance", "self-sufficiency", "self-reliance", "self-direction"],
    "mystery": ["mystery", "enigma", "secret", "riddle", "unknown", "veil", "shadow", "cryptic", "puzzle", "obscure", 
                "hidden", "arcane", "unfathomable", "esoteric", "clandestine", "intrigue", "uncertain", "mystic", 
                "enigmatic", "concealed", "occult", "inscrutable", "perplexing", "baffling", "paradox", "conundrum",
                "ambiguity", "uncertainty", "doubt", "question", "query", "uncertainty", "vagueness"],
    "spirituality": ["spirit", "soul", "divine", "sacred", "holy", "eternal", "faith", "grace", "heaven", "prayer", 
                     "meditation", "transcend", "blessing", "reverence", "devotion", "enlightenment", "cosmos", 
                     "serenity", "awe", "sanctuary", "deity", "god", "goddess", "divinity", "transcendence",
                     "nirvana", "zen", "karma", "dharma", "moksha", "salvation", "redemption", "afterlife"]
}

BADGE_CRITERIA = {
    "The Visionary": {
        "weights": {
            "metaphor_count": 0.25,
            "personification_count": 0.2,
            "surprise_score": 0.2,
            "nature_theme": 0.15,
            "imagery_score": 0.2
        },
        "description": "Weaves vivid imagery and profound insights with imaginative flair."
    },
    "The Passionate": {
        "weights": {
            "anger_score": 0.25,
            "happy_score": 0.2,
            "love_score": 0.2,
            "struggle_theme": 0.2,
            "hyperbole_count": 0.15
        },
        "description": "Burns with intense emotions and raw, powerful expression."
    },
    "The Reflective": {
        "weights": {
            "sadness_score": 0.25,
            "time_theme": 0.25,
            "identity_theme": 0.2,
            "oxymoron_count": 0.15,
            "diversity": 0.15
        },
        "description": "Explores deep introspection and philosophical musings."
    },
    "The Hopeful": {
        "weights": {
            "hopeful_score": 0.3,
            "hope_theme": 0.25,
            "positive_sentiment": 0.2,
            "alliteration_count": 0.15,
            "freedom_theme": 0.1
        },
        "description": "Radiates optimism and inspires with uplifting messages."
    }
}

ARCHETYPE_KEYWORDS = {
    "hero's journey": ["journey", "quest", "path", "trial", "victory", "triumph", "overcome", "challenge", "adventure", 
                       "odyssey", "mission", "destiny", "hero", "courage", "epic", "expedition", "valor", "questing", 
                       "ordeal", "conquest", "bravery", "venture", "pilgrimage", "crusade", "campaign", "voyage",
                       "exploration", "search", "pursuit", "undertaking", "enterprise", "initiative"],
    "fall from grace": ["fall", "ruin", "downfall", "regret", "sorrow", "decline", "shame", "loss", "failure", "descent", 
                        "collapse", "tragedy", "demise", "doom", "disgrace", "plummet", "wreck", "undoing", "calamity", 
                        "catastrophe", "misfortune", "reversal", "degeneration", "deterioration", "decay", "degradation",
                        "abasement", "humiliation", "abasement", "mortification", "abasement"],
    "coming of age": ["grow", "learn", "discover", "youth", "mature", "change", "become", "realize", "awakening", 
                      "initiation", "rite", "transition", "bloom", "evolve", "transform", "passage", "enlighten", 
                      "emerge", "self-discovery", "maturity", "revelation", "development", "progress", "advancement",
                      "evolution", "metamorphosis", "transformation", "flowering", "ripening", "fruition"],
    "rebirth": ["renew", "rebirth", "rise", "dawn", "awaken", "revive", "renewal", "resurrection", "renaissance", 
                "revitalize", "regenerate", "restore", "reawaken", "redeem", "revival", "phoenix", "rejuvenate", 
                "metamorphosis", "born anew", "salvation", "recovery", "resurgence", "reanimation", "reinvigoration",
                "rekindling", "resuscitation", "reawakening", "rejuvenation", "reconstitution", "reestablishment"],
    "tragic love": ["love", "heartbreak", "parting", "sacrifice", "doomed", "forbidden", "passion", "loss", "yearning", 
                    "separation", "devotion", "tragedy", "unrequited", "fate", "longing", "anguish", "betrayal", 
                    "destined", "ill-fated", "sorrowful", "star-crossed", "fatal", "ruinous", "catastrophic",
                    "disastrous", "calamitous", "hapless", "wretched", "miserable", "pitiable"],
    "quest for truth": ["truth", "search", "seek", "unveil", "reveal", "mystery", "enlighten", "discover", "clarity", 
                        "wisdom", "insight", "epiphany", "pursuit", "quest", "illumination", "revelation", "uncover", 
                        "knowledge", "understanding", "disclosure", "investigation", "inquiry", "examination",
                        "exploration", "research", "study", "analysis", "scrutiny", "inspection", "probe"]
}

# Enhanced helper functions
def get_random_quote():
    return random.choice(QUOTES)

def calculate_lexical_diversity(poem_text):
    try:
        words = [word.lower() for word in re.findall(r'\b\w+\b', poem_text) if len(word) > 2]
        total_words = len(words)
        unique_words = len(set(words))
        diversity = unique_words / total_words if total_words else 0
        
        if diversity > 0.75:
            desc = "Extraordinary lexical richness with diverse vocabulary"
            expl = "The poem employs a wide-ranging vocabulary, minimizing repetition for nuanced expression."
        elif diversity > 0.55:
            desc = "Robust lexical diversity with varied word choice"
            expl = "The poem balances repetition for effect with diverse words for depth."
        elif diversity > 0.35:
            desc = "Moderate lexical diversity with strategic repetition"
            expl = "The poem uses repetition intentionally while maintaining variety."
        else:
            desc = "Focused vocabulary with strong repetition"
            expl = "The poem leverages repetition for rhythmic or thematic emphasis."
            
        return diversity, total_words, unique_words, desc, expl
    except Exception as e:
        logger.error(f"Error in lexical diversity: {e}")
        return 0.0, 0, 0, "N/A", "Error assessing vocabulary variety."
def count_syllables_total(poem_text):
    try:
        words = re.findall(r'\b\w+\b', poem_text)
        total = sum(syllables.estimate(word) for word in words)
        explanation = (
            f"Total syllables: {total}. "
            f"Estimated using syllable counting rules for rhythm analysis."
        )
        return total, explanation
    except Exception as e:
        logger.error(f"Error counting syllables: {e}")
        return 0, "Error counting syllables"

def get_synonyms(word, limit=20):
    try:
        synonyms = set()
        for syn in wordnet.synsets(word.lower()):
            for lemma in syn.lemmas():
                if lemma.name().lower() != word.lower():
                    synonyms.add(lemma.name().lower().replace('_', ' '))
            for hyper in syn.hypernyms():
                for lemma in hyper.lemmas():
                    synonyms.add(lemma.name().lower().replace('_', ' '))
        return sorted(synonyms, key=lambda x: len(x))[:limit]
    except Exception as e:
        logger.error(f"Error fetching synonyms: {e}")
        return []

def get_antonyms(word, limit=15):
    try:
        antonyms = set()
        for syn in wordnet.synsets(word.lower()):
            for lemma in syn.lemmas():
                for ant in lemma.antonyms():
                    if ant.name().lower() != word.lower():
                        antonyms.add(ant.name().lower())
        return sorted(antonyms, key=lambda x: len(x))[:limit]
    except Exception as e:
        logger.error(f"Error fetching antonyms: {e}")
        return []

def hex_to_rgb(hex_color):
    try:
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    except Exception as e:
        logger.error(f"Error converting hex to RGB: {e}")
        return (1.0, 1.0, 1.0)

def rgb_to_hex(rgb):
    try:
        return '#{:02x}{:02x}{:02x}'.format(
            int(max(0, min(1, rgb[0])) * 255),
            int(max(0, min(1, rgb[1])) * 255),
            int(max(0, min(1, rgb[2])) * 255)
        )
    except Exception as e:
        logger.error(f"Error converting RGB to hex: {e}")
        return '#FFFFFF'

def generate_tint_or_shade(base_color, factor):
    try:
        rgb = hex_to_rgb(base_color)
        h, l, s = colorsys.rgb_to_hls(*rgb)
        l = max(0.05, l / (factor * 1.3)) if factor > 1 else min(0.98, l + (1 - l) * (1 - factor) * 1.3)
        new_rgb = colorsys.hls_to_rgb(h, l, s)
        return rgb_to_hex(new_rgb)
    except Exception as e:
        logger.error(f"Error generating tint/shade: {e}")
        return base_color

def generate_color_palette(base_color, num_colors=6):
    try:
        rgb = hex_to_rgb(base_color)
        h, l, s = colorsys.rgb_to_hls(*rgb)
        palette = [base_color]
        for i in range(1, num_colors):
            factor = 1 + (i * 0.25) if i % 2 == 0 else 1 - (i * 0.2)
            new_rgb = colorsys.hls_to_rgb(h, max(0.05, min(0.95, l * factor)), s)
            palette.append(rgb_to_hex(new_rgb))
        explanation = (
            f"Generated {num_colors} colors harmonizing with {base_color}. "
            f"Tints and shades adjust lightness for emotional resonance."
        )
        return palette, explanation
    except Exception as e:
        logger.error(f"Error generating color palette: {e}")
        return [base_color] * num_colors, "Error generating color palette."

# Enhanced analysis functions 
def detect_figurative_language(poem_text):
    try:
        lines = [line.strip() for line in poem_text.splitlines() if line.strip()]
        figures = []
        
        # Enhanced patterns 
        simile_markers = [r"\b(like|as if|as though|as|similar to|resembles|akin to|comparable to|much like|just as|so as|as much as|in the manner of|in the way that|in the fashion of)\b"]
        metaphor_markers = [r"\b(is|are|was|were|be|become|seems|appears|represents|symbolizes|stands for|embodies|personifies|epitomizes|exemplifies|typifies|characterizes|constitutes)\b"]
        personification_verbs = [
            "sings", "dances", "whispers", "shouts", "cries", "laughs", "weeps", "dreams", "speaks", "breathes",
            "runs", "waits", "hopes", "thinks", "feels", "watches", "yearns", "muses", "prays", "sighs", "longs",
            "wanders", "gazes", "smiles", "embraces", "mourns", "rejoices", "trembles", "soars", "rests", "awakens",
            "whimpers", "giggles", "contemplates", "ponders", "reflects", "meditates", "considers", "deliberates",
            "whistles", "groans", "moans", "howls", "chuckles", "snickers", "grumbles", "murmurs", "stammers"
        ]
        hyperbole_patterns = [
            r"\b(?:millions?|billions?|thousands of|never ending|eternity|forever|infinitely|beyond measure|countless|unimaginable|all the \w+ in the world|always|never)\b",
            r"\b(?:so \w+ that|died \w+|raining \w+ and \w+|heart stopped|world \w+ed|sky fell|\d+\s*(times|fold)|every \w+ in existence|universe \w+ed)\b",
            r"\b(?:infinite|endless|limitless|immeasurable|boundless|incalculable|colossal|astronomical|monumental|gargantuan|prodigious|stupendous|tremendous|herculean|titanic)\b"
        ]
        oxymoron_patterns = [
            r"\b(?:bittersweet|deafening silence|living dead|seriously funny|awfully good|cruel kindness|silent scream|jumbo shrimp|sweet sorrow|open secret|organized chaos|alone together|dark light|old news|only choice|original copy|painfully beautiful|small crowd|true myth|wise fool|bitter joy|gentle storm)\b",
            r"\b(?:fierce calm|quiet roar|loud whisper|bright shadow|soft thunder|heavy lightness|cold fire|warm ice|sweet pain|freezing fire|burning cold|dark sun|bright night|joyful sadness|peaceful war|orderly chaos|random order|definite maybe|exact estimate|found missing)\b"
        ]
        onomatopoeia_words = [
            "buzz", "crash", "whisper", "bang", "sizzle", "roar", "hiss", "boom", "clap", "murmur", "ding", "swoosh",
            "thud", "pop", "crackle", "whoosh", "tick", "tock", "splash", "growl", "meow", "woof", "chirp", "rustle",
            "clatter", "hum", "drip", "slap", "whack", "thump", "clang", "jingle", "rattle", "snap", "creak", "moan",
            "clink", "plop", "drizzle", "patter", "thunder", "whimper", "gurgle", "squelch", "whizz", "zap", "vroom"
        ]
        alliteration_pattern = r'\b(\w)\w*\s+\1\w*(?:\s+\1\w*)?\b'
        assonance_pattern = r'\b\w*([aeiou])\w*\s+\w*\1\w*\b'
        consonance_pattern = r'\b\w*([bcdfghjklmnpqrstvwxyz])\w*\s+\w*\1\w*\b'
        imagery_indicators = ["vivid", "gleam", "shimmer", "sparkle", "glow", "radiance", "hue", "shade", "color", 
                             "vision", "sight", "glimpse", "tableau", "panorama", "scene", "portrait", "glint", 
                             "luster", "scintilla", "illumination", "brilliance", "luminosity", "sheen", "polish",
                             "glitter", "twinkle", "glisten", "shining", "dazzle", "iridescence"]

        vader = SentimentIntensityAnalyzer()
        if not nlp or not embedder:
            logger.warning("Missing Spacy or SentenceTransformer")
            return figures, "Cannot analyze figurative language due to missing models."

        for line_idx, line in enumerate(lines):
            lower_line = line.lower().strip()
            if len(lower_line.split()) < 2:
                continue
                
            doc = nlp(line)
            tokens = [token.text.lower() for token in doc]
            pos_tags = [(token.text, token.pos_) for token in doc]
            has_noun = any(pos == 'NOUN' for _, pos in pos_tags)
            sentiment_score = vader.polarity_scores(line)['compound']
            embeddings = embedder.encode([line])[0]
            context_keywords = [word for word in tokens if word in sum([EMOTION_KEYWORDS[e]["keywords"] for e in EMOTION_KEYWORDS], [])]
            stanza = '\n\n'.join(lines[max(0, line_idx-2):line_idx+3])
            stanza_emb = embedder.encode([stanza])[0]

            if not has_noun:
                continue

            # Enhanced simile detection
            for marker in simile_markers:
                if re.search(marker, lower_line, re.IGNORECASE):
                    for i, token in enumerate(doc):
                        if token.dep_ in ["nsubj", "nsubjpass"] and i + 2 < len(doc) and doc[i + 2].dep_ in ["dobj", "attr", "pobj"]:
                            emb1 = embedder.encode([token.text])[0]
                            emb2 = embedder.encode([doc[i + 2].text])[0]
                            sim = cosine_similarity([emb1], [emb2])[0][0]
                            context_sim = cosine_similarity([embeddings], [stanza_emb])[0][0]
                            conf = min(0.99, 0.9 + (context_sim * 0.15) + (0.1 if context_keywords else 0) + (0.05 if sim < 0.5 else 0))
                            explanation = (
                                f"Compares '{token.text}' to '{doc[i + 2].text}' using '{marker.lower()}'. "
                                f"Semantic distance: {sim:.2f}. "
                                f"{'Emotionally charged' if context_keywords else 'Descriptive comparison'}. "
                                f"Analyzed with Spacy, dependency parsing, and SentenceTransformer."
                            )
                            figures.append(("Simile", line, conf, explanation))
                            break

            # Enhanced metaphor detection
            for marker in metaphor_markers:
                if re.search(marker, lower_line, re.IGNORECASE) and not any(w in lower_line for w in ["like", "as", "similar to"]):
                    for i, token in enumerate(doc):
                        if token.pos_ == "NOUN" and i + 2 < len(doc) and tokens[i + 1].lower() in ["is", "are", "was", "were", "be", "match", "seems", "appears"] and doc[i + 2].pos_ in ["NOUN", "ADJ"]:
                            lemma1, lemma2 = token.lemma_.lower(), doc[i + 2].lemma_.lower()
                            is_abstract = any(
                                syn.hypernyms() and any(hypo.name().split('.')[0] in ['abstraction', 'concept', 'attribute', 'emotion']
                                for lemma in [lemma1, lemma2] for syn in wordnet.synsets(lemma)
                            ))
                            emb1 = embedder.encode([lemma1])[0]
                            emb2 = embedder.encode([lemma2])[0]
                            sim = cosine_similarity([emb1], [emb2])[0][0]
                            conf = min(0.99, 0.95 + (0.1 if is_abstract else 0) + (0.08 if abs(sentiment_score) > 0.6 else 0))
                            explanation = (
                                f"Equates '{lemma1}' with '{lemma2}' metaphorically. "
                                f"Semantic distance: {sim:.2f}. "
                                f"{'Abstract concept' if is_abstract else 'Concrete imagery'}. "
                                f"Analyzed with WordNet, sentiment, and SentenceTransformer."
                            )
                            figures.append(("Metaphor", line, conf, explanation))
                            break

            # Enhanced personification detection
            for token in doc:
                if token.pos_ == "NOUN" and token.head.lemma_.lower() in personification_verbs:
                    is_nonhuman = any(
                        syn.hypernyms() and any(hypo.name().split('.')[0] in ['object', 'natural_object', 'phenomenon', 'plant', 'animal']
                        for hypo in syn.hypernyms())
                        for syn in wordnet.synsets(token.lemma_.lower())
                    )
                    if is_nonhuman:
                        conf = min(0.99, 0.93 + (0.15 if sentiment_score != 0 else 0) + (0.08 if context_keywords else 0))
                        explanation = (
                            f"Animates '{token.text}' with '{token.head.text}'. "
                            f"{'Emotionally evocative' if sentiment_score != 0 else 'Vivid imagery'}. "
                            f"Analyzed with Spacy, WordNet, and VADER."
                        )
                        figures.append(("Personification", line, conf, explanation))
                        break

            # Enhanced hyperbole detection
            if any(re.search(pat, lower_line, re.IGNORECASE) for pat in hyperbole_patterns) or abs(sentiment_score) > 0.8:
                conf = min(0.99, 0.91 + (0.15 if abs(sentiment_score) > 0.8 else 0))
                matched = next((pat for pat in hyperbole_patterns if re.search(pat, lower_line, re.IGNORECASE)), "extreme sentiment")
                explanation = (
                    f"Exaggerates via '{matched}'. "
                    f"Sentiment intensity: {abs(sentiment_score):.2f}. "
                    f"Analyzed with VADER and regex patterns."
                )
                figures.append(("Hyperbole", line, conf, explanation))

            # Enhanced oxymoron detection
            if any(re.search(pat, lower_line, re.IGNORECASE) for pat in oxymoron_patterns):
                conf = 0.99
                matched = next((pat for pat in oxymoron_patterns if re.search(pat, lower_line, re.IGNORECASE)), "")
                explanation = (
                    f"Oxymoron '{matched}' creates a paradoxical effect. "
                    f"Analyzed with regex."
                )
                figures.append(("Oxymoron", line, conf, explanation))
            else:
                for i, token in enumerate(doc[:-1]):
                    if token.pos_ in ["ADJ", "NOUN"] and doc[i + 1].pos_ in ["ADJ", "NOUN"]:
                        word1, word2 = token.lemma_.lower(), doc[i + 1].lemma_.lower()
                        if word2 in get_antonyms(word1) or word1 in get_antonyms(word2):
                            conf = 0.98
                            explanation = (
                                f"Oxymoron '{word1} {word2}' evokes contradiction. "
                                f"Analyzed with WordNet antonyms."
                            )
                            figures.append(("Oxymoron", line, conf, explanation))
                            break

            # Enhanced alliteration detection
            if re.search(alliteration_pattern, lower_line):
                words = re.findall(r'\b\w+\b', lower_line)
                for i in range(len(words) - 2):
                    phones1 = pronouncing.phones_for_word(words[i])[0].split() if pronouncing.phones_for_word(words[i]) else []
                    phones2 = pronouncing.phones_for_word(words[i + 1])[0].split() if pronouncing.phones_for_word(words[i + 1]) else []
                    if phones1 and phones2 and phones1[0] == phones2[0]:
                        conf = 0.98
                        explanation = (
                            f"Alliteration in '{words[i]}' and '{words[i+1]}'. "
                            f"Enhances rhythmic flow. Analyzed with pronouncing."
                        )
                        figures.append(("Alliteration", line, conf, explanation))
                        break
                    elif i < len(words) - 1 and words[i][0].lower() == words[i + 1][0].lower() and len(words[i]) > 2:
                        conf = 0.96
                        explanation = (
                            f"Alliteration in '{words[i]}' and '{words[i+1]}'. "
                            f"Adds musicality. Analyzed with regex."
                        )
                        figures.append(("Alliteration", line, conf, explanation))
                        break

            # Enhanced onomatopoeia detection
            if any(w in tokens for w in onomatopoeia_words):
                conf = 0.97
                matched = next((w for w in onomatopoeia_words if w in tokens), "sound effect")
                explanation = (
                    f"'{matched}' mimics natural sounds for effect. "
                    f"Analyzed with regex."
                )
                figures.append(("Onomatopoeia", line, conf, explanation))

            # Enhanced assonance detection
            if re.search(assonance_pattern, lower_line):
                conf = 0.95
                matched = re.findall(assonance_pattern, lower_line)[0]
                explanation = (
                    f"Assonance with vowel '{matched}'. "
                    f"Creates melodic resonance. Analyzed with regex."
                )
                figures.append(("Assonance", line, conf, explanation))

            # Enhanced consonance detection
            if re.search(consonance_pattern, lower_line):
                conf = 0.95
                matched = re.findall(consonance_pattern, lower_line)[0]
                explanation = (
                    f"Consonance with '{matched}'. "
                    f"Reinforces rhythmic structure. Analyzed with regex."
                )
                figures.append(("Consonance", line, conf, explanation))

            # Enhanced imagery detection
            if any(w in tokens for w in imagery_indicators) or any(token.text.lower() in sum([THEME_KEYWORDS["nature"], THEME_KEYWORDS["mystery"]], []) for token in doc):
                conf = min(0.98, 0.88 + (0.15 if context_keywords else 0) + (0.08 if abs(sentiment_score) > 0.5 else 0))
                explanation = (
                    f"Evokes vivid sensory imagery in '{line.strip()}'. "
                    f"{'Emotionally charged' if context_keywords else 'Descriptive'}. "
                    f"Analyzed with Spacy and keyword matching."
                )
                figures.append(("Imagery", line, conf, explanation))

        explanation_summary = (
            "Figurative language detected using Spacy, SentenceTransformer, VADER, WordNet, pronouncing, and regex. "
            "Confidence scores reflect syntactic, semantic, and contextual analysis."
        )
        return sorted(set(figures), key=lambda x: x[2], reverse=True)[:20], explanation_summary
    except Exception as e:
        logger.error(f"Error in figurative language detection: {e}")
        return [], "Error detecting figurative language."

def analyze_sentiment(poem_text):
    try:
        blob = TextBlob(poem_text)
        vader = SentimentIntensityAnalyzer()
        vader_scores = vader.polarity_scores(poem_text)
        doc = nlp(poem_text) if nlp else None
        
        # Multi-model sentiment aggregation
        sentiment_score = vader_scores['compound'] * 0.45
        if doc:
            sentiment_score += sum(token.sentiment for token in doc if token.sentiment) / (len([t for t in doc]) or 1) * 0.35
        sentiment_score += blob.sentiment.polarity * 0.2
        
        subjectivity_score = blob.sentiment.subjectivity * 0.7
        if doc:
            subjectivity_score += (sum(1 if t.dep_ in ['amod', 'advmod'] else 0 for t in doc) / len(doc)) * 0.3
        
        tone = (
            "Highly Positive" if sentiment_score > 0.4 else
            "Positive" if sentiment_score > 0.1 else
            "Highly Negative" if sentiment_score < -0.4 else
            "Negative" if sentiment_score < -0.1 else
            "Neutral"
        )
        tone_desc = (
            "radiant and uplifting" if sentiment_score > 0.4 else
            "optimistic and warm" if sentiment_score > 0.1 else
            "dark and anguished" if sentiment_score < -0.4 else
            "somber and reflective" if sentiment_score < -0.1 else
            "balanced and nuanced"
        )
        explanation = (
            f"Exudes a {tone_desc} tone. "
            f"Subjectivity ({subjectivity_score:.2f}) reflects {'emotional depth' if subjectivity_score > 0.6 else 'factual clarity' if subjectivity_score < 0.4 else 'balanced expression'}. "
            f"Computed using VADER (45%), Spacy (35%), TextBlob (20%)."
        )
        return sentiment_score, subjectivity_score, tone, explanation
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return 0.0, 0.0, "Neutral", "Error assessing sentiment."

def get_top_words(poem_text, top_n=20):
    try:
        print(f"Original text: {poem_text}")  # Debug
        words = re.findall(r'\b\w+\b', poem_text.lower())
        print(f"All words: {words}")  # Debug
        
        # Try to load NLTK stopwords, fall back to a default list if unavailable
        try:
            stop_words = set(nltk.corpus.stopwords.words('english') + ['the', 'a', 'an', 'with', 'by'])
        except LookupError:
            logger.warning("NLTK stopwords not found, using fallback stopwords")
            stop_words = set([
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 
                'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 
                'will', 'with', 'by'
            ])
        
        filtered_words = [w for w in words if w not in stop_words and len(w) > 2 and w.isalpha()]
        print(f"Filtered words: {filtered_words}")  # Debug
        word_freq = Counter(filtered_words)
        print(f"Word frequencies: {word_freq}")  # Debug
        top_words = word_freq.most_common(top_n) if word_freq else []
        explanation = (
            f"Identified {len(top_words)} top words after filtering stopwords and short words. "
            f"Computed using Counter and {'NLTK stopwords' if 'nltk' in locals() else 'fallback stopwords'}."
        )
        return top_words, word_freq, explanation
    except Exception as e:
        print(f"Error in get_top_words: {e}")  # Debug
        logger.error(f"Error in get_top_words: {e}")
        return [], Counter(), f"Error analyzing word frequency: {str(e)}"

def get_readability(poem_text):
    try:
        score = textstat.flesch_reading_ease(poem_text)
        levels = [
            (95, "4th–5th grade"), (90, "5th–6th grade"), (80, "7th grade"), (70, "8th–9th grade"),
            (60, "10th–11th grade"), (50, "12th grade"), (30, "College"), (0, "Advanced")
        ]
        level = next(lvl for thr, lvl in levels if score >= thr)
        explanation = (
            f"{'Highly accessible' if score >= 90 else 'Moderately complex' if score >= 60 else 'Intellectually demanding' if score >= 30 else 'Highly sophisticated'} for {level} readers. "
            f"Based on sentence length and syllable complexity. Computed with textstat."
        )
        return score, level, explanation
    except Exception as e:
        logger.error(f"Error in readability: {e}")
        return None, "N/A", "Error assessing readability."
def get_emotions_nrc(text):
    try:
        text_object = NRCLex(text)
        raw_scores = text_object.raw_emotion_scores
        mapped_scores = {emo: 0.0 for emo in EMOTION_KEYWORDS}
        for nrc_emotion, score in raw_scores.items():
            target_emotion = NRCLEX_TO_EMOTION.get(nrc_emotion)
            if target_emotion in mapped_scores:
                mapped_scores[target_emotion] += score * 1.5
        total = sum(mapped_scores.values()) or 1
        for emo in mapped_scores:
            mapped_scores[emo] /= total
        explanation = (
            f"Normalized emotion scores derived from NRCLex. "
            f"Reflects emotional nuances with enhanced weighting."
        )
        return mapped_scores, explanation
    except Exception as e:
        logger.error(f"Error in NRCLex emotion analysis: {e}")
        return {emo: 0.0 for emo in EMOTION_KEYWORDS}, "Error analyzing emotions."
def recompute_for_lyrics(text, emotion_keys):
    try:
        emotion_vector = {emo: 0.0 for emo in emotion_keys}
        words = re.findall(r'\b\w+\b', text.lower())
        total_words = len(words) or 1
        for emo, data in EMOTION_KEYWORDS.items():
            keyword_hits = sum(1 for w in words if w in data['keywords'])
            emotion_vector[emo] += (keyword_hits / total_words) * 3.0  # Stronger keyword emphasis
        if embedder:
            text_emb = embedder.encode([text])[0]
            emotion_centroids = {
                emo: np.mean([embedder.encode([k])[0] for k in EMOTION_KEYWORDS[emo]['keywords'][:3]], axis=0)
                for emo in emotion_keys
            }
            for emo, centroid in emotion_centroids.items():
                sim = cosine_similarity([text_emb], [centroid])[0][0]
                if sim > 0.75:
                    emotion_vector[emo] += sim * 0.7
        vader_scores = vader.polarity_scores(text)
        combined_polarity = vader_scores['compound']
        if combined_polarity > 0.4:
            emotion_vector['happy'] += 0.3
            emotion_vector['hopeful'] += 0.2
        elif combined_polarity < -0.4:
            emotion_vector['sadness'] += 0.3
            emotion_vector['fear'] += 0.2
        total = sum(emotion_vector.values()) or 1
        for emo in emotion_vector:
            emotion_vector[emo] /= total
        logger.info(f"Recomputed lyric vector: {emotion_vector}")
        return emotion_vector
    except Exception as e:
        logger.error(f"Error in recompute_for_lyrics: {e}")
        return {emo: 1.0 / len(emotion_keys) for emo in emotion_keys}
def analyze_emotions_comprehensive(poem_text):
    """
    Analyzes emotions in a poem with high precision, prioritizing primary emotions and restricting niche ones.
    Returns a normalized emotion vector and a concise explanation.
    """
    try:
        logger.info("Starting enhanced emotion analysis")
        emotions = ['happy', 'sadness', 'anger', 'calm', 'fear', 'surprise', 'hopeful', 'disgust']
        emotion_vector = {emo: 0.0 for emo in emotions}
        
        # Strict emotion relationships
        EMOTION_SYNERGIES = {
            ('happy', 'hopeful'): 0.2,
            ('happy', 'calm'): 0.15,
            ('sadness', 'fear'): 0.2,
            ('hopeful', 'calm'): 0.15
        }
        EMOTION_CONFLICTS = {
            'happy': ['sadness', 'fear'],
            'sadness': ['happy', 'calm'],
            'anger': ['happy', 'calm'],
            'calm': ['anger', 'fear', 'surprise'],
            'fear': ['happy', 'calm', 'hopeful'],
            'hopeful': ['sadness', 'fear'],
            'surprise': ['calm'],
            'disgust': ['happy', 'calm', 'hopeful']
        }
        
        # Initialize tools
        blob = TextBlob(poem_text)
        vader = SentimentIntensityAnalyzer()
        doc = nlp(poem_text) if nlp else None
        words = re.findall(r'\b\w+\b', poem_text.lower())
        total_words = len(words) or 1
        sentences = list(blob.sentences)
        total_sentences = len(sentences) or 1

        # 1. Sentence-level analysis with strict thresholds
        sentence_scores = []
        for idx, sentence in enumerate(sentences):
            sent_text = str(sentence)
            sent_doc = nlp(sent_text) if doc else None
            sent_words = re.findall(r'\b\w+\b', sent_text.lower())
            sent_score = {emo: 0.0 for emo in emotions}

            # 1.1 Transformer-based classification with high confidence
            if emotion_classifier:
                results = emotion_classifier(sent_text)[0]
                label_mapping = {
                'joy': 'happy', 'sadness': 'sadness', 'anger': 'anger',
                'fear': 'fear', 'surprise': 'surprise', 'disgust': 'disgust',
                'neutral': 'calm'
            }
            for result in results:
                label = result['label'].lower()
                score = result['score']
                if label in label_mapping and score > 0.7:  # Strict threshold
                    emo = label_mapping[label]
                    weight = 0.6 if emo in ['anger', 'disgust'] else 0.8 if emo != 'calm' else 0.4  # Lower weight for calm
                    sent_score[emo] += score * weight

            # 1.2 Semantic similarity with emotion centroids
            if embedder and sent_words:
                sent_emb = embedder.encode([sent_text])[0]
                emotion_centroids = {
                    emo: np.mean([embedder.encode([k])[0] for k in EMOTION_KEYWORDS[emo]['keywords'][:3]], axis=0)
                    for emo in emotions
                }
                for emo, centroid in emotion_centroids.items():
                    sim = cosine_similarity([sent_emb], [centroid])[0][0]
                    if sim > 0.85:  # Very strict threshold
                        weight = 0.3 if emo in ['anger', 'disgust'] else 0.5
                        sent_score[emo] += sim * weight
                        if sent_doc and any(token.dep_ in ['amod', 'advmod'] for token in sent_doc):
                            sent_score[emo] *= 1.1

            # 1.3 Sentiment-based adjustments
            vader_scores = vader.polarity_scores(sent_text)
            blob_polarity = sentence.sentiment.polarity
            combined_polarity = (vader_scores['compound'] * 0.65 + blob_polarity * 0.35)
            intensity = abs(combined_polarity) * 0.3
            if combined_polarity > 0.3:
                sent_score['happy'] += intensity
                sent_score['hopeful'] += intensity * 0.7
            elif combined_polarity < -0.3:
                sent_score['sadness'] += intensity
                sent_score['fear'] += intensity * 0.6

            # 1.4 Precise keyword matching
            for emo, data in EMOTION_KEYWORDS.items():
                keyword_hits = sum(1 for w in sent_words if w in data['keywords'])
                if keyword_hits and emo not in ['anger', 'disgust']:
                    sent_score[emo] += (keyword_hits / len(sent_words)) * 0.25
                elif keyword_hits and emo in ['anger', 'disgust']:
                    if keyword_hits / len(sent_words) > 0.2:  # Require strong presence
                        sent_score[emo] += (keyword_hits / len(sent_words)) * 0.15

            # 1.5 Apply synergies and conflicts
            for (emo1, emo2), boost in EMOTION_SYNERGIES.items():
                if sent_score[emo1] > 0.3 and sent_score[emo2] > 0.3:
                    sent_score[emo1] += boost * sent_score[emo2]
                    sent_score[emo2] += boost * sent_score[emo1]
            for emo, conflicts in EMOTION_CONFLICTS.items():
                if sent_score[emo] > 0.3:
                    for conflict in conflicts:
                        if sent_score[conflict] > 0:
                            reduction = min(sent_score[conflict], sent_score[emo] * 0.4)
                            sent_score[conflict] -= reduction

            # Normalize sentence scores
            sent_sum = max(sum(sent_score.values()), 0.1)  # Ensure minimum sum to avoid over-normalization
            for emo in sent_score:
                sent_score[emo] = max(0, sent_score[emo] / sent_sum)
            sentence_scores.append(sent_score)

        # 2. Aggregate with positional weighting
        for idx, sent_score in enumerate(sentence_scores):
            weight = 1.3 if idx in [0, len(sentence_scores) - 1] else 1.0
            if doc:
                sent_text = str(sentences[idx])
                sent_doc = nlp(sent_text)
                if any(token.dep_ in ['nsubj', 'dobj', 'root'] for token in sent_doc):
                    weight *= 1.2
            for emo in emotions:
                emotion_vector[emo] += sent_score[emo] * weight
        total_weight = sum(1.3 if i in [0, len(sentence_scores) - 1] else 1.0 for i in range(len(sentence_scores))) or 1
        for emo in emotions:
            emotion_vector[emo] /= total_weight

        # 3. NRCLex for context
        try:
            text_object = NRCLex(poem_text)
            raw_scores = text_object.raw_emotion_scores
            for nrc_emotion, score in raw_scores.items():
                target_emotion = NRCLEX_TO_EMOTION.get(nrc_emotion)
                if target_emotion in emotions and target_emotion not in ['anger', 'disgust']:
                    emotion_vector[target_emotion] += score * 0.2
                elif target_emotion in ['anger', 'disgust'] and score > 0.5:  # Strict threshold
                    emotion_vector[target_emotion] += score * 0.1
        except Exception as e:
            logger.warning(f"NRCLex error: {e}")

        # 4. Semantic coherence with poem-level embeddings
        if embedder:
            poem_emb = embedder.encode([poem_text])[0]
            emotion_profiles = {
                emo: np.mean([embedder.encode([k])[0] for k in EMOTION_KEYWORDS[emo]['keywords'][:3]], axis=0)
                for emo in emotions
            }
            for emo, profile in emotion_profiles.items():
                sim = cosine_similarity([poem_emb], [profile])[0][0]
                if sim > 0.9:  # Extremely strict
                    weight = 0.15 if emo in ['anger', 'disgust'] else 0.25
                    emotion_vector[emo] *= (1 + sim * weight)

        # 5. Normalize with tempered softmax
        scores = np.array([emotion_vector[emo] for emo in emotions])
        temperature = 0.8  # Moderate temperature for focus
        exp_scores = np.exp(scores / temperature)
        softmax_scores = exp_scores / np.sum(exp_scores)
        for idx, emo in enumerate(emotions):
            emotion_vector[emo] = float(softmax_scores[idx])

        # 6. Enforce niche emotion restrictions
        for emo in ['anger', 'disgust']:
            if emotion_vector[emo] > 0.15:
                excess = emotion_vector[emo] - 0.15
                emotion_vector[emo] = 0.15
                emotion_vector['calm'] += excess * 0.5  # Redistribute to calm
                emotion_vector['sadness'] += excess * 0.5  # Redistribute to sadness

        # 7. Final output
        dominant_emotion = max(emotion_vector, key=emotion_vector.get)
        confidence = emotion_vector[dominant_emotion]
        sorted_emotions = sorted(emotion_vector.items(), key=lambda x: x[1], reverse=True)
        explanation = (
            f"Dominant: {dominant_emotion.capitalize()} ({confidence:.2f}). "
            f"Top emotions: {', '.join(f'{emo.capitalize()} ({score:.2f})' for emo, score in sorted_emotions[:3])}. "
            f"Analyzed via transformers, embeddings, and sentiment with strict thresholds."
        )

        logger.info(f"Emotion analysis complete: {emotion_vector}")
        return emotion_vector, explanation

    except Exception as e:
        logger.error(f"Error in emotion analysis: {str(e)}", exc_info=True)
        neutral_scores = {emo: 0.0 for emo in emotions}
        neutral_scores['calm'] = 1.0
        return neutral_scores, "Error analyzing emotions - defaulting to calm."

def generate_wordcloud(poem_text, dominant_emotion):
    try:
        # Try to load NLTK stopwords, fall back to a default list if unavailable
        try:
            stop_words = set(nltk.corpus.stopwords.words('english') + ['the', 'a', 'an', 'with', 'by'])
        except LookupError:
            logger.warning("NLTK stopwords not found, using fallback stopwords")
            stop_words = set([
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 
                'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 
                'will', 'with', 'by'
            ])
        
        # Generate word cloud
        wc = WordCloud(
            width=800, height=400, background_color='#1a1a1a',
            colormap='viridis', stopwords=stop_words, min_font_size=10
        ).generate(poem_text)
        explanation = (
            f"Word cloud generated with {len(stop_words)} stopwords filtered. "
            f"Colored to reflect {dominant_emotion} using 'viridis' colormap."
        )
        return wc, explanation
    except Exception as e:
        logger.error(f"Error generating word cloud: {e}")
        return None, f"Error generating word cloud: {str(e)}"

    
def vector_to_array(vector, keys):
    try:
        return np.array([vector.get(k, 0.0) for k in keys])
    except Exception as e:
        logger.error(f"Error in vector conversion: {e}")
        return np.zeros(len(keys))

global_json_path = None
def load_song_data():
    global global_json_path
    for path in possible_paths:
        try:
            logger.info(f"Attempting to load song data from: {path}")
            
            # Try standard json.load first (works for smaller files)
            try:
                with open(path, "r", encoding='utf-8') as f:
                    songs_db = json.load(f)
                logger.info(f"Successfully loaded {len(songs_db)} songs using standard JSON parser")
                global_json_path = path
                return songs_db
            except (MemoryError, json.JSONDecodeError) as e:
                logger.warning(f"Standard JSON load failed (file may be too large), trying streaming parser: {str(e)}")
            
            # Fall back to ijson streaming parser for large files
            try:
                import ijson
                songs_db = []
                with open(path, "r", encoding='utf-8') as f:
                    # Parse the file incrementally
                    parser = ijson.items(f, 'item')  # Assumes JSON structure is a top-level array
                    for song in parser:
                        if isinstance(song, dict):  # Basic validation
                            songs_db.append(song)
                
                logger.info(f"Successfully loaded {len(songs_db)} songs using streaming parser")
                global_json_path = path
                return songs_db
            except Exception as e:
                logger.error(f"Streaming parser failed: {str(e)}")
                continue
                
        except Exception as e:
            logger.error(f"Error loading {path}: {str(e)}")
            continue
    
    logger.error("All loading attempts failed - no valid song data found")
    return []  # Return empty list if all attempts fail

def generate_playlist(poem_emotion_vector, songs_db, top_n=5):
    global global_json_path
    try:
        if not songs_db:
            logger.error("No song data available in database")
            return [], "No song data available in database."
        
        if not poem_emotion_vector:
            logger.error("No emotion vector provided for comparison")
            return [], "Could not analyze poem's emotions."
            
        emotion_keys = list(EMOTION_KEYWORDS.keys())
        if len(emotion_keys) != 8:
            logger.error("Mismatch between emotion keys and vector size")
            return [], "Configuration error in emotion analysis."
            
        target_vector = np.array(list(poem_emotion_vector.values())).reshape(1, -1)
        
        if np.all(target_vector == 0):
            logger.error("Emotion vector is all zeros")
            return [], "Could not determine poem's emotional profile."
        
        scored_songs = []
        valid_songs = 0
        updated_songs = False
        
        # Resolve json_path
        json_path = None
        for path in possible_paths:
            try:
                if not path:
                    logger.warning("Empty path in possible_paths, skipping")
                    continue
                dir_path = os.path.dirname(path) or os.getcwd()
                if os.path.exists(path) and os.access(path, os.W_OK):
                    json_path = path
                    logger.info(f"Found writable existing path: {json_path}")
                    break
                if os.access(dir_path, os.W_OK):
                    json_path = path
                    logger.info(f"Confirmed writable directory for: {json_path}")
                    break
            except Exception as e:
                logger.warning(f"Path {path} not usable: {str(e)}")
                continue
        
        # Fallback to temporary directory
        if not json_path:
            temp_dir = tempfile.gettempdir()
            json_path = os.path.join(temp_dir, "songs.json")
            logger.info(f"No valid path found, using temporary path: {json_path}")
        
        # Ensure songs_db is a list
        if not isinstance(songs_db, list):
            logger.error("songs_db is not a list")
            songs_db = []
            return [], "Invalid song database format."
        
        for song in songs_db:
            logger.info(f"Processing song: {song.get('title', 'Unknown')}")
            if not song.get('lyrics') or len(song['lyrics']) < 10:
                logger.warning(f"Skipping song {song.get('title', 'Unknown')} due to insufficient lyrics")
                continue
                
            valid_songs += 1
            lyrics = song['lyrics']
            
            try:
                # Check if song has a valid emotion vector
                if ('emotion_vector' in song and 
                    isinstance(song['emotion_vector'], list) and 
                    len(song['emotion_vector']) == 8 and
                    sum(1 for score in song['emotion_vector'] if score > 0) >= 2):
                    song_vector = dict(zip(emotion_keys, song['emotion_vector']))
                    logger.info(f"Using existing emotion vector for song: {song.get('title', 'Unknown')}")
                else:
                    song_vector, _ = analyze_emotions_comprehensive(lyrics)
                    # Check if vector is overly skewed
                    max_score = max(song_vector.values())
                    dominant_emotion = max(song_vector, key=song_vector.get)
                    if max_score > 0.9 and dominant_emotion == 'calm':
                        logger.warning(f"Overly skewed emotion vector for {song.get('title', 'Unknown')}, using uniform vector")
                        song_vector = {emo: 1.0 / len(emotion_keys) for emo in emotion_keys}
                    song['emotion_vector'] = [song_vector.get(emotion, 0) for emotion in emotion_keys]
                    logger.info(f"Updated emotion vector for {song.get('title', 'Unknown')}: {song['emotion_vector']}")
                    updated_songs = True

                song_array = np.array([song_vector.get(emotion, 0) for emotion in emotion_keys]).reshape(1, -1)
                emotion_sim = cosine_similarity(target_vector, song_array)[0][0]
                
                text_sim = 0
                if embedder:
                    try:
                        poem_emb = embedder.encode([poem])[0]
                        lyrics_emb = embedder.encode([lyrics])[0]
                        text_sim = cosine_similarity([poem_emb], [lyrics_emb])[0][0]
                    except Exception as e:
                        logger.warning(f"Error in text embedding comparison: {e}")
                
                combined_score = emotion_sim * 0.7 + text_sim * 0.3
                scored_songs.append((combined_score, song))
                
            except Exception as e:
                logger.warning(f"Error processing song {song.get('title', 'Unknown')}: {e}")
                continue
        
        # Always attempt to write to json_path for debugging
        logger.info(f"Attempting to save songs_db to {json_path} (updated_songs: {updated_songs}, valid_songs: {valid_songs})")
        try:
            dir_path = os.path.dirname(json_path) or os.getcwd()
            os.makedirs(dir_path, exist_ok=True)
            with open(json_path, "w", encoding='utf-8') as f:
                json.dump(songs_db, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully saved songs_db to {json_path}")
            global_json_path = json_path
        except Exception as e:
            logger.error(f"Failed to save songs_db to {json_path}: {str(e)}")
            st.error(f"Failed to save songs.json: {str(e)}")
            return [], f"Failed to save songs.json: {str(e)}"
        
        if not scored_songs:
            logger.error("No songs could be scored successfully")
            return [], "No matching songs could be analyzed."
            
        scored_songs.sort(reverse=True, key=lambda x: x[0])
        top_songs = [s[1] for s in scored_songs[:top_n]]
        
        explanation = (
            f"Found {len(top_songs)} emotionally matching songs. "
            f"Top matches: {', '.join([song.get('title', 'Unknown') for song in top_songs[:2]])}. "
            f"Matching based on emotional profile (70%) and lyrical content (30%)."
        )
        
        return top_songs, explanation
        
    except Exception as e:
        logger.error(f"Error generating playlist: {str(e)}")
        st.error(f"Error generating playlist: {str(e)}")
        return [], "Error generating playlist."
   
def categorize_poem_length(total_words):
    try:
        if total_words < 30:
            return "Short", "A concise burst of poetic insight."
        elif total_words < 100:
            return "Medium", "A balanced canvas for depth and nuance."
        elif total_words < 250:
            return "Long", "A deep narrative with layered complexity."
        else:
            return "Epic", "A grand tapestry of poetic exploration."
    except Exception as e:
        logger.error(f"Error in poem length: {e}")
        return "N/A", "Error categorizing length."
    
def analyze_themes(poem_text):
    """
    Analyzes a poem to detect themes with high precision and stability, prioritizing lighter themes (love, loss, hope, struggle, time)
    and restricting heavier themes (spirituality, identity, freedom). Nature is favored as a secondary theme unless overwhelmingly dominant.
    Returns theme scores and an explanation with theme descriptions.
    """
    try:
        # Define theme opposites for strong conflict resolution
        THEME_OPPOSITES = {
            'love': 'loss',
            'loss': 'love',
            'hope': 'struggle',
            'struggle': 'hope',
            'identity': 'mystery',
            'mystery': 'identity',
            'freedom': None,  # Neutral, no direct opposite
            'nature': None,   # Neutral, no direct opposite
            'time': None,     # Neutral, no direct opposite
            'spirituality': None  # Neutral, no direct opposite
        }
        
        # Define theme descriptions
        THEME_DESCRIPTIONS = {
            'nature': 'Natural beauty, landscapes, and elemental harmony',
            'love': 'Romantic affection, connection, and emotional intimacy',
            'loss': 'Loss, mourning, and emotional absence',
            'hope': 'Optimism, aspiration, and forward-looking dreams',
            'time': 'Temporal flow, memory, and the passage of eras',
            'struggle': 'Conflict, hardship, and perseverance',
            'identity': 'Self-discovery, individuality, and personal essence',
            'freedom': 'Liberty, autonomy, and unshackled spirit',
            'mystery': 'Enigma, secrecy, and the unknown',
            'spirituality': 'Divine connection, faith, and transcendence'
        }
        
        # Classify themes
        LIGHT_THEMES = ['love', 'loss', 'hope', 'struggle', 'time']
        HEAVY_THEMES = ['spirituality', 'identity', 'freedom']
        
        # Initialize theme vector
        theme_vector = {theme: 0.0 for theme in THEME_KEYWORDS}
        words = re.findall(r'\b\w+\b', poem_text.lower())
        total_words = len(words) or 1  # Avoid division by zero
        
        # Initialize analysis tools
        blob = TextBlob(poem_text)
        doc = nlp(poem_text) if nlp else None

        # 1. Precise keyword-based scoring with TF-IDF
        pseudo_corpus = [poem_text] + [' '.join(keywords) for keywords in THEME_KEYWORDS.values()]
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, norm='l2', use_idf=True, min_df=1)
        tfidf_matrix = tfidf_vectorizer.fit_transform(pseudo_corpus)
        tfidf_scores = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))

        for theme, keywords in THEME_KEYWORDS.items():
            # Limit synonyms to 1 per keyword, only for top 2 keywords
            synonyms = []
            for keyword in keywords[:2]:
                try:
                    synonyms.extend(get_synonyms(keyword, 1))
                except:
                    continue
            all_keywords = list(set(keywords + synonyms))
            
            for word in words:
                lemma = Word(word).lemmatize()
                if lemma in [Word(k).lemmatize() for k in all_keywords]:
                    base_score = 7.0 if lemma in [Word(k).lemmatize() for k in keywords] else 1.5  # Stronger penalty for synonyms
                    if theme in HEAVY_THEMES:
                        base_score *= 0.5  # Restrict heavy themes
                    elif theme == 'nature':
                        base_score *= 0.8  # Slightly reduce nature
                    tfidf_weight = tfidf_scores.get(lemma, 0.05) * 1.0  # Minimal TF-IDF weight
                    score = base_score * (1 + tfidf_weight)
                    if doc:
                        for token in doc:
                            if token.lemma_.lower() == lemma:
                                if theme in HEAVY_THEMES and token.dep_ not in ['root', 'nsubj']:
                                    score *= 0.5  # Heavy themes require strong syntactic roles
                                elif token.dep_ in ['nsubj', 'dobj', 'root']:
                                    score *= 1.4  # Boost for light themes in key roles
                    theme_vector[theme] += score / total_words
        logger.info(f"Post-TF-IDF theme scores: {theme_vector}")

        # 2. Strict semantic similarity for theme keywords
        if embedder:
            theme_keyword_embeddings = {
                theme: np.mean([embedder.encode([k])[0] for k in keywords[:4]], axis=0)
                for theme, keywords in THEME_KEYWORDS.items()
            }
            word_embeddings = {word: embedder.encode([word])[0] for word in set(words) if len(word) > 3}
            for word in word_embeddings:
                word_emb = word_embeddings[word]
                for theme, keyword_emb in theme_keyword_embeddings.items():
                    sim = cosine_similarity([word_emb], [keyword_emb])[0][0]
                    if sim > 0.95:  # Extremely strict threshold
                        score = sim * 1.0 / total_words  # Minimal contribution
                        if theme in HEAVY_THEMES:
                            score *= 0.4  # Restrict heavy themes
                        elif theme == 'nature':
                            score *= 0.7  # Reduce nature
                        if doc:
                            for token in doc:
                                if token.lemma_.lower() == word and token.dep_ in ['nsubj', 'dobj']:
                                    score *= 1.2  # Minimal boost
                        theme_vector[theme] += score
        logger.info(f"Post-semantic similarity theme scores: {theme_vector}")

        # 3. Focused sentence-level contextual analysis
        for sentence in blob.sentences:
            sentence_text = str(sentence)
            sent_doc = nlp(sentence_text) if doc else None
            sent_words = [Word(w).lemmatize() for w in re.findall(r'\b\w+\b', sentence_text.lower())]
            
            for theme, keywords in THEME_KEYWORDS.items():
                keyword_set = set([Word(k).lemmatize() for k in keywords])  # Exact matches only
                if any(w in keyword_set for w in sent_words):
                    score = 1.0 / total_words  # Minimal score
                    if theme in HEAVY_THEMES:
                        score *= 0.5  # Restrict heavy themes
                    elif theme == 'nature':
                        score *= 0.8  # Slightly reduce nature
                    if sent_doc:
                        for token in sent_doc:
                            if token.lemma_.lower() in keyword_set and token.dep_ in ['nsubj', 'dobj', 'root']:
                                score *= 1.5  # Stronger boost for key roles
                    theme_vector[theme] += score
                    if theme in THEME_OPPOSITES and THEME_OPPOSITES[theme]:
                        opposite = THEME_OPPOSITES[theme]
                        theme_vector[opposite] = max(0, theme_vector[opposite] - score * 0.8)  # Strong reduction
        logger.info(f"Post-sentence-level theme scores: {theme_vector}")

        # 4. Minimal co-occurrence analysis for light themes only
        theme_pairs = list(combinations(LIGHT_THEMES + ['nature', 'mystery'], 2))  # Exclude heavy themes
        for sentence in blob.sentences:
            sent_words = [Word(w).lemmatize() for w in re.findall(r'\b\w+\b', str(sentence).lower())]
            for (theme1, theme2) in theme_pairs:
                theme1_keywords = set([Word(k).lemmatize() for k in THEME_KEYWORDS[theme1]])
                theme2_keywords = set([Word(k).lemmatize() for k in THEME_KEYWORDS[theme2]])
                if any(w in theme1_keywords for w in sent_words) and any(w in theme2_keywords for w in sent_words):
                    theme_vector[theme1] += 0.1 / total_words  # Minimal boost
                    theme_vector[theme2] += 0.1 / total_words
                elif sent_doc:
                    if not any(w in theme1_keywords for w in sent_words):
                        theme_vector[theme1] = max(0, theme_vector[theme1] - 0.2 / total_words)  # Stronger penalty
                    if not any(w in theme2_keywords for w in sent_words):
                        theme_vector[theme2] = max(0, theme_vector[theme2] - 0.2 / total_words)
        logger.info(f"Post-cooccurrence theme scores: {theme_vector}")

        # 5. Nature suppression unless overwhelmingly dominant
        max_score = max(theme_vector.values()) or 1
        if theme_vector['nature'] < 2.0 * max([theme_vector[t] for t in LIGHT_THEMES]):  # Nature needs 2x lead
            theme_vector['nature'] *= 0.6  # Suppress nature to secondary

        # 6. Aggressive normalization for distinctiveness
        for theme in theme_vector:
            theme_vector[theme] = (theme_vector[theme] / max_score) ** 3  # Cubic scaling
            if theme in THEME_OPPOSITES and THEME_OPPOSITES[theme]:
                opposite = THEME_OPPOSITES[theme]
                reduction = theme_vector[theme] * 0.8  # Aggressive reduction
                theme_vector[opposite] = max(0, theme_vector[opposite] - reduction)
            if theme in HEAVY_THEMES:
                theme_vector[theme] *= 0.4  # Final restriction for heavy themes

        # Softmax with very low temperature for sharp distribution
        temperature = 0.3  # Very low for distinctiveness
        scores = np.array(list(theme_vector.values()))
        exp_scores = np.exp(scores / temperature)
        softmax_scores = exp_scores / np.sum(exp_scores)
        softmax_scores = np.clip(softmax_scores, 0, 1)

        for i, theme in enumerate(theme_vector.keys()):
            theme_vector[theme] = float(softmax_scores[i])
        logger.info(f"Final theme scores: {theme_vector}")

        # Stability check: Ensure dominant theme is consistent across poem segments
        dominant_theme = max(theme_vector.items(), key=lambda x: x[1])[0]
        segment_scores = {theme: 0.0 for theme in THEME_KEYWORDS}
        for sentence in blob.sentences[:3]:  # Check first 3 sentences for stability
            sent_doc = nlp(str(sentence)) if doc else None
            sent_words = [Word(w).lemmatize() for w in re.findall(r'\b\w+\b', str(sentence).lower())]
            sent_vector = {theme: 0.0 for theme in THEME_KEYWORDS}
            for theme, keywords in THEME_KEYWORDS.items():
                keyword_set = set([Word(k).lemmatize() for k in keywords])
                if any(w in keyword_set for w in sent_words):
                    score = 1.0
                    if theme in HEAVY_THEMES:
                        score *= 0.5
                    elif theme == 'nature':
                        score *= 0.8
                    sent_vector[theme] += score
            sent_max = max(sent_vector.values()) or 1
            for theme in sent_vector:
                sent_vector[theme] /= sent_max
            sent_dominant = max(sent_vector.items(), key=lambda x: x[1])[0]
            segment_scores[sent_dominant] += 1
        if segment_scores[dominant_theme] < max(segment_scores.values()):
            # If dominant theme isn't consistent, boost the most frequent segment theme
            new_dominant = max(segment_scores.items(), key=lambda x: x[1])[0]
            if new_dominant in LIGHT_THEMES or new_dominant == 'nature':
                theme_vector[new_dominant] *= 1.5
                max_score = max(theme_vector.values()) or 1
                for theme in theme_vector:
                    theme_vector[theme] /= max_score
                exp_scores = np.exp(np.array(list(theme_vector.values())) / temperature)
                softmax_scores = exp_scores / np.sum(exp_scores)
                for i, theme in enumerate(theme_vector.keys()):
                    theme_vector[theme] = float(softmax_scores[i])
        logger.info(f"Post-stability check theme scores: {theme_vector}")

        # Get dominant theme and secondary themes
        dominant_theme = max(theme_vector.items(), key=lambda x: x[1])[0]
        confidence = theme_vector[dominant_theme]
        secondary_themes = [(theme, score) for theme, score in theme_vector.items() if score > 0.05 and theme != dominant_theme]
        
        # Build explanation with theme descriptions
        explanation = (
            f"Dominant theme: {dominant_theme.capitalize()} (confidence: {confidence:.2f}) - {THEME_DESCRIPTIONS.get(dominant_theme, 'No description available')}.\n"
        )
        if secondary_themes:
            explanation += "Secondary themes:\n"
            for theme, score in sorted(secondary_themes, key=lambda x: x[1], reverse=True):
                explanation += f"- {theme.capitalize()} (confidence: {score:.2f}) - {THEME_DESCRIPTIONS.get(theme, 'No description available')}.\n"
        explanation += (
            "Analysis uses precise TF-IDF matching, strict semantic similarity, focused sentence context, "
            "minimal co-occurrence for light themes, aggressive normalization, and stability checks for a clear dominant theme."
        )
        
        return theme_vector, explanation

    except Exception as e:
        logger.error(f"Error in theme analysis: {str(e)}", exc_info=True)
        neutral_scores = {theme: 0.0 for theme in THEME_KEYWORDS}
        neutral_scores['love'] = 1.0  # Default to love for poem context
        return neutral_scores, "Error analyzing themes - defaulting to love."
def escape_latex(text):
    try:
        latex_special_chars = {
            '#': r'\#', '$': r'\$', '%': r'\$', '&': r'\&&', '_': r'\_',
            '{': r'\{', '}': r'\}', '~': r'\textasciitilde', '^': r'\textasciicircum}',
            '\\': r'\textbackslash', '<': r'\textless', '>': r'\textgreater'
        }
        return ''.join(latex_special_chars.get(c, c) for c in text)
    except Exception as e:
        logger.error(f"Error escaping LaTeX: {e}")
        return text

def generate_analysis_report(poem_text, sentiment, subjectivity, mood, reading_score, reading_level, emotion_scores, total_syllables, poem_length, themes, theme_scores, figures, badge, archetype, themes_desc):
    try:
        report = StringIO()
        report.write("📜 Mumbai Poets Society Analysis Report\n")
        report.write("=====================================\n\n")
        report.write(f"**Mood**: {mood}\n")
        report.write(f"**Sentiment Score**: {sentiment:.3f} (positive > 0.1, negative < -0.1)\n")
        report.write(f"**Subjectivity Score**: {subjectivity:.3f} (0=objective, 1=subjective)\n")
        report.write(f"**Readability Score**: {reading_score if reading_score else 'N/A'} (higher = easier)\n")
        report.write(f"**Suggested Reading Level**: {reading_level}\n")
        report.write(f"**Total Syllables**: {total_syllables}\n")
        report.write(f"**Length Category**: {poem_length}\n")
        report.write(f"**Dominant Themes**: {', '.join(themes_desc)}\n")
        report.write(f"**Poetic Badge**: {badge}\n")
        report.write(f"**Narrative Archetype**: {archetype}\n\n")
        report.write("**Emotion Breakdown**:\n")
        for emotion, score in sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True):
            report.write(f" - {emotion.capitalize()} {EMOTION_KEYWORDS[emotion]['emoji']}: {score:.3f}\n")
        report.write("\n**Theme Scores**:\n")
        for theme, score in sorted(theme_scores.items(), key=lambda x: x[1], reverse=True):
            report.write(f" - {theme.capitalize()}: {score:.3f}\n")
        report.write("\n**Literary Devices Detected**:\n")
        for fig_type, line, conf, expl in sorted(figures, key=lambda x: x[2], reverse=True):
            report.write(f" - **{fig_type}** (Confidence: {conf:.3f}): {line.strip()}\n")
            report.write(f"   *Explanation*: {expl}\n")
        report.write("\n**Original Poem**:\n")
        report.write(poem_text)
        report.write('\n')
        return report.getvalue()
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return "Error generating report."

def generate_latex_report(poem_text, sentiment, subjectivity, mood, reading_score, reading_level, emotion_scores, total_syllables, poem_length, themes, theme_scores, figures, badge, archetype):
    try:
        escaped_poem = escape_latex(poem_text)
        escaped_mood = escape_latex(mood)
        escaped_reading_level = escape_latex(reading_level)
        escaped_poem_length = escape_latex(poem_length)
        escaped_themes = escape_latex(', '.join(themes))
        escaped_badge = escape_latex(badge)
        escaped_archetype = escape_latex(archetype)
        escaped_emotions = ''.join([f'\\item {emotion.capitalize()} {escape_latex(EMOTION_KEYWORDS[emotion]["emoji"])}: {score:.3f}' for emotion, score in sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)])
        escaped_theme_scores = ''.join([f'\\item {theme.capitalize()}: {score:.3f}' for theme, score in sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)])
        escaped_figures = ''.join([
            f'\\item \\textbf{{{fig_type.capitalize()}}} (Confidence: {conf:.3f}): \\newline {escape_latex(line.strip())} \\\\ '
            f'\\textit{{Explanation: {escape_latex(expl)}}}\\\\'
            for fig_type, line, conf, expl in sorted(figures, key=lambda x: x[2], reverse=True)
        ])

        # Preprocess the poem to replace newlines with LaTeX double backslashes
        latex_formatted_poem = escaped_poem.replace('\n', '\\\\')

        latex_content = f"""
\\documentclass{{article}}
\\usepackage{{geometry}}
\\geometry{{a4paper, margin=1in}}
\\usepackage{{utf8x}}
\\usepackage{{parskip}}
\\usepackage{{enumitem}}
\\usepackage{{titlesec}}
\\usepackage{{emoji}}
\\usepackage{{amsmath}}
\\usepackage{{fontspec}}
\\setmainfont{{Noto Serif}}
\\titleformat{{\\section}}{{\\Large\\bfseries}}{{\\thesection}}{{1em}}{{}}
\\begin{{document}}
\\section*{{Mumbai Poets Society Poetry Analysis Report}}
\\textbf{{Mood:}} {escaped_mood}\\\\
\\textbf{{Sentiment Score:}} {sentiment:.3f} (positive > 0.1, negative < -0.1)\\\\
\\textbf{{Subjectivity Score:}} {subjectivity:.3f} (0=objective, 1=subjective)\\\\
\\textbf{{Readability Score:}} {reading_score if reading_score else 'N/A'} (higher = easier)\\\\
\\textbf{{Suggested Reading Level:}} {escaped_reading_level}\\\\
\\textbf{{Total Syllables:}} {total_syllables}\\\\
\\textbf{{Poem Length:}} {escaped_poem_length}\\\\
\\textbf{{Dominant Themes:}} {escaped_themes}\\\\
\\textbf{{Poetic Badge:}} {escaped_badge}\\\\
\\textbf{{Narrative Archetype:}} {escaped_archetype}\\\\
\\section*{{Emotion Breakdown}}
\\begin{{itemize}}
{escaped_emotions}
\\end{{itemize}}
\\section*{{Theme Scores}}
\\begin{{itemize}}
{escaped_theme_scores}
\\end{{itemize}}
\\section*{{Literary Devices Detected}}
\\begin{{itemize}}
{escaped_figures}
\\end{{itemize}}
\\section*{{Original Poem}}
\\begin{{verse}}
{latex_formatted_poem}
\\end{{verse}}
\\end{{document}}
"""
        return latex_content
    except Exception as e:
        logger.error(f"Error generating LaTeX report: {e}")
        return "\\documentclass{article}\\begin{document}Error generating report.\\end{document}"
def generate_poetic_badge(diversity, figures, themes, theme_scores, emotion_scores):
    try:
        figure_counts = Counter(fig_type for fig_type, _, _, _ in figures)
        sentiment_score = sum(emotion_scores[e] for e in ["happy", "hopeful", "calm", "surprise"]) - \
                         (sum(emotion_scores[e] for e in ["sadness", "anger", "fear", "disgust"]))
        badge_scores = {}
        
        for badge, criteria in BADGE_CRITERIA.items():
            score = 0.0
            weights = criteria["weights"]
            if badge == "The Visionary":
                score += figure_counts.get("Metaphor", 0) * weights["metaphor_count"]
                score += figure_counts.get("Personification", 0) * weights["personification_count"]
                score += emotion_scores.get("surprise", 0) * weights["surprise_score"]
                score += theme_scores.get("nature", 0) * weights["nature_theme"]
                score += figure_counts.get("Imagery", 0) * weights["imagery_score"]
            elif badge == "The Passionate":
                score += emotion_scores.get("anger", 0) * weights["anger_score"]
                score += emotion_scores.get("happy", 0) * weights["happy_score"]
                score += theme_scores.get("struggle", 0) * weights["struggle_theme"]
                score += figure_counts.get("Hyperbole", 0) * weights["hyperbole_count"]
            elif badge == "The Reflective":
                score += emotion_scores.get("sadness", 0) * weights["sadness_score"]
                score += theme_scores.get("time", 0) * weights["time_theme"]
                score += theme_scores.get("identity", 0) * weights["identity_theme"]
                score += figure_counts.get("Oxymoron", 0) * weights["oxymoron_count"]
                score += (1 - diversity) * weights["diversity"]
            elif badge == "The Hopeful":
                score += emotion_scores.get("hopeful", 0) * weights["hopeful_score"]
                score += theme_scores.get("hope", 0) * weights["hope_theme"]
                score += max(0, sentiment_score) * weights["positive_sentiment"]
                score += figure_counts.get("Alliteration", 0) * weights["alliteration_count"]
                score += theme_scores.get("freedom", 0) * weights["freedom_theme"]
            badge_scores[badge] = score

        badge = max(badge_scores, key=badge_scores.get) if max(badge_scores.values()) > 0.6 else "The Eclectic"
        explanation = (
            f"Awarded '{badge}' badge. "
            f"{BADGE_CRITERIA.get(badge, {'description': 'Versatile and distinctive voice'})['description']}. "
            f"Derived from diversity ({diversity:.2f}), literary devices, themes, and emotions."
        )
        return badge, explanation
    except Exception as e:
        logger.error(f"Error generating badge: {e}")
        return "N/A", "Error generating badge."
def classify_narrative_archetype(poem_text, themes, theme_scores, emotion_scores, sentiment):
    try:
        stanzas = [s.strip() for s in poem_text.split('\n\n') if s.strip()]
        archetype_scores = {arch: 0.0 for arch in ARCHETYPE_KEYWORDS}
        words = re.findall(r'\b\w+\b', poem_text.lower())
        doc = nlp(poem_text) if nlp else None
        expanded_keywords = {
            arch: keywords + sum([get_synonyms(word, 10) for word in keywords[:5]], [])
            for arch, keywords in ARCHETYPE_KEYWORDS.items()
        }

        # Keyword-based scoring
        for arch, keywords in expanded_keywords.items():
            keyword_count = sum(words.count(k) * 3.0 for k in keywords)
            if doc:
                for token in doc:
                    if token.lemma_.lower() in keywords and token.dep_ in ["nsubj", "dobj", "root"]:
                        keyword_count += 2.0
            archetype_scores[arch] = keyword_count

        # Embedding-based similarity
        if embedder:
            poem_emb = embedder.encode([poem_text])[0]
            for arch in ARCHETYPE_KEYWORDS:
                arch_emb = np.mean([embedder.encode([k])[0] for k in ARCHETYPE_KEYWORDS[arch]], axis=0)
                sim = cosine_similarity([poem_emb], [arch_emb])[0][0]
                if sim > 0.65:
                    archetype_scores[arch] += sim * 3.0

        # Contextual scoring based on themes and emotions
        if "struggle" in themes or theme_scores.get("struggle", 0) > 0.25:
            archetype_scores["hero's journey"] += 0.5
        if "loss" in themes or theme_scores.get("loss", 0) > 0.25:
            archetype_scores["fall from grace"] += 0.6
        if "hope" in themes or theme_scores.get("hope", 0) > 0.25:
            archetype_scores["coming of age"] += 0.5
            archetype_scores["rebirth"] += 0.5
        if "mystery" in themes or theme_scores.get("mystery", 0) > 0.25:
            archetype_scores["quest for truth"] += 0.7

        # Emotional context scoring
        if emotion_scores.get("hopeful", 0) > 0.3 and sentiment > 0.2:
            archetype_scores["hero's journey"] += 0.4
            archetype_scores["rebirth"] += 0.3
        if emotion_scores.get("sadness", 0) > 0.3 and sentiment < -0.2:
            archetype_scores["fall from grace"] += 0.5
        if emotion_scores.get("confusion", 0) > 0.3:
            archetype_scores["quest for truth"] += 0.5

        # Structural scoring
        if len(stanzas) > 3:
            archetype_scores["hero's journey"] += 0.3
            archetype_scores["quest for truth"] += 0.2
        elif len(stanzas) == 1:
            archetype_scores["fall from grace"] += 0.3

        archetype = max(archetype_scores, key=archetype_scores.get) if max(archetype_scores.values()) > 0.8 else "Undetermined"
        
        archetype_descriptions = {
            "hero's journey": "Follows a protagonist's transformative quest through challenges to triumph.",
            "fall from grace": "Depicts a tragic descent from prosperity to ruin or despair.",
            "coming of age": "Chronicles personal growth and self-discovery.",
            "rebirth": "Portrays renewal, redemption, or spiritual awakening.",
            "tragic love": "Explores doomed or unrequited romantic passion.",
            "quest for truth": "Seeks enlightenment or revelation of hidden knowledge.",
            "Undetermined": "No clear narrative pattern detected."
        }
        
        explanation = (
            f"Identified as '{archetype.capitalize()}'. "
            f"{archetype_descriptions.get(archetype.lower(), 'Unique narrative structure')}. "
            f"Determined through keyword analysis, semantic similarity, and contextual scoring."
        )
        return archetype, explanation
    except Exception as e:
        logger.error(f"Error classifying archetype: {e}")
        return "Undetermined", "Error classifying archetype."

def generate_summary(poem, sentiment, mood, themes, emotion_scores, badge, archetype):
    try:
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        summary = StringIO()
        summary.write("✨ Mumbai Poets Society Analysis Summary\n")
        summary.write("=====================================\n")
        summary.write(f"Mood: {mood} (Score: {sentiment:.3f})\n")
        summary.write(f"Dominant Emotion: {dominant_emotion.capitalize()} {EMOTION_KEYWORDS[dominant_emotion]['emoji']} ({emotion_scores[dominant_emotion]:.3f})\n")
        summary.write(f"Key Themes: {', '.join(themes[:3])}\n")
        summary.write(f"Poetic Style: {badge}\n")
        summary.write(f"Narrative Pattern: {archetype}\n")
        summary.write("\nPoem Excerpt:\n")
        lines = poem.strip().splitlines()
        excerpt = '\n'.join(lines[:min(4, len(lines))]) + ('...' if len(lines) > 4 else '')
        summary.write(excerpt)
        return summary.getvalue()
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return "Error generating summary."
#Loading the animations
lottie_plane = load_lottie("Paper Plane.json")
lottie_grammy = load_lottie("Grammy.json")
lottie_hourglass = load_lottie("Hourglass.json")
lottie_search = load_lottie("Search animation.json")
lottie_type=load_lottie("typewriter.json")
lottie_dots=load_lottie("dots.json")
lottie_book=load_lottie("book.json")
loading_message = st.empty()

# Initialize session state
if 'writing_prompt' not in st.session_state:
    st.session_state.writing_prompt = None
if 'analysis_id' not in st.session_state:
    st.session_state.analysis_id = str(uuid.uuid4())

# Main App
st.title("📜 Mumbai Poets Society Poetry Analyzer")
st.markdown(f"*{get_random_quote()}*")
st.markdown("Discover the hidden dimensions of your poetry through advanced linguistic analysis.")

# Custom CSS for enhanced UI
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom, #1a1a1a, #2c2c2c);
        color: #ffffff;
    }
    .stTextArea textarea {
        background-color: #2c2c2c;
        color: #ffffff;
        border: 1px solid #FFD700;
        border-radius: 8px;
    }
    .stButton button {
        background-color: #FFD700;
        color: #1a1a1a;
        border-radius: 8px;
        transition: all 0.3s;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #66B2FF;
        transform: scale(1.05);
    }
    .stExpander {
        background-color: #2c2c2c;
        border: 1px solid #6B7280;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .stMetric {
        background-color: #3a3a3a;
        border-radius: 8px;
        padding: 10px;
    }
    .report-title {
        color: #FFD700;
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    .expander-description {
        color: #AAAAAA;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Poet's Toolkit")
    if st.button("✨ Generate Writing Prompt"):
        prompts = [
            "A whisper carried by the monsoon winds",
            "The last lantern glowing in a forgotten alley",
            "A heart stitched together by starlight",
            "Shadows dancing on a crumbling wall",
            "A river's secrets told at midnight",
            "The weight of a dream too heavy to hold",
            "Time unraveling in a lover's gaze",
            "The color of loneliness",
            "A map drawn with disappearing ink",
            "The scent of rain on parched earth",
            "A letter never sent",
            "Footsteps echoing in an empty corridor",
            "The silence between two heartbeats",
            "A feather floating in a storm",
            "The last page of a well-worn book"
        ]
        st.session_state.writing_prompt = random.choice(prompts)
    if st.session_state.writing_prompt:
        st.info(f"**Prompt**: {st.session_state.writing_prompt}")
    st.markdown("---")
    st.markdown("**About Mumbai Poets Society**")
    st.markdown("Celebrating the art of poetry through analysis and inspiration.")

# Poem Input
poem = st.text_area(
    "Enter your poem for analysis:",
    height=300,
    placeholder="Paste your poem here...",
    key="poem_input"
)


# Analysis
if st.button("Analyze Your Poem"):
    if not poem.strip():
        st.error("Please enter a poem to analyze")
    else:
        st.session_state.analysis_id = str(uuid.uuid4())
        logger.info(f"Starting analysis for session {st.session_state.analysis_id}")
        progress = st.progress(0)
        steps = 10
        step = 0
        st_lottie(lottie_dots, height=200)
        # Sentiment Analysis
        step += 1
        progress.progress(min(step / steps, 1.0))
        sentiment, subjectivity, mood, sent_expl = analyze_sentiment(poem)
        with st.expander("🧠 Sentiment Analysis", expanded=True):
            st.markdown("<div class='expander-description'>Understand the emotional tone and subjectivity of your poem</div>", unsafe_allow_html=True)
            st.markdown(f"**Tone**: {mood} ({sentiment:.2f})")
            st.markdown(f"**Subjectivity**: {subjectivity:.2f}")
            st.markdown(f"*{sent_expl}*")


        # Readability
        step += 1
        progress.progress(min(step / steps, 1.0))
        reading_score, reading_level, read_expl = get_readability(poem)
        with st.expander("📚 Readability"):
            st.markdown("<div class='expander-description'>Assess how accessible your poem is to different reading levels</div>", unsafe_allow_html=True)
            if reading_score:
                st.markdown(f"**Score**: {reading_score:.1f}")
                st.markdown(f"**Level**: {reading_level}")
                st.markdown(f"*{read_expl}*")
            else:
                st.error("Unable to assess readability")

        # Lexical Diversity
        step += 1
        progress.progress(min(step / steps, 1.0))
        diversity, total_words, unique_words, div_desc, div_expl = calculate_lexical_diversity(poem)
        with st.expander("🧬 Lexical Diversity"):
            st.markdown("<div class='expander-description'>Measure the richness and variety of your vocabulary</div>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Words", total_words)
            col2.metric("Unique Words", unique_words)
            col3.metric("Diversity", f"{diversity:.2f}")
            st.markdown(f"**{div_desc}**")
            st.markdown(f"*{div_expl}*")

        st_lottie(lottie_plane, height=200)

        # Emotion Analysis
        step += 1
        progress.progress(min(step / steps, 1.0))
        emotion_scores, emotion_expl = analyze_emotions_comprehensive(poem)
        dominant_emotion = max(emotion_scores, key=emotion_scores.get) if emotion_scores else "calm"

        # Emotion Visualizations
        step += 1
        progress.progress(min(step / steps, 1.0))
        with st.expander("🌈 Emotional Landscape"):
            st.markdown("<div class='expander-description'>Explore the emotional profile of your poem</div>", unsafe_allow_html=True)
            if emotion_scores:
                emotion_df = pd.DataFrame.from_dict(emotion_scores, orient='index', columns=['Score'])
                color_map = {emo: EMOTION_KEYWORDS[emo.lower()]["color"] for emo in emotion_df.index}
                
                # Bar Chart
                fig_bar = px.bar(
                    emotion_df, x=emotion_df.index, y='Score', title="Emotion Intensity",
                    color=emotion_df.index, color_discrete_map=color_map,
                    text=emotion_df['Score'].apply(lambda x: f"{x:.2f}")
                )
                fig_bar.update_layout(
                    plot_bgcolor='#1a1a1a', paper_bgcolor='#1a1a1a', font_color='#FFFFFF',
                    xaxis_title="Emotions", yaxis_title="Intensity",
                    bargap=0.10,  # Reduce gap to make bars thicker, like buildings
                )
                st.plotly_chart(fig_bar, use_container_width=False)
                # Radar Chart
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=list(emotion_scores.values()),
                    theta=list(emotion_scores.keys()),
                    fill='toself',
                    name='Emotion Profile',
                    line_color=EMOTION_KEYWORDS[dominant_emotion]["color"]
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, max(1.0, max(emotion_scores.values()) * 1.2)]),
                        bgcolor='#1a1a1a'
                    ),
                    showlegend=True,
                    title="Emotion Radar",
                    paper_bgcolor='#1a1a1a',
                    font_color='#FFFFFF'
                )
                st.plotly_chart(fig_radar, use_container_width=False)

                st.markdown(f"**Dominant Emotion**: {dominant_emotion.capitalize()} {EMOTION_KEYWORDS[dominant_emotion]['emoji']} ({emotion_scores[dominant_emotion]:.2f})")
                st.markdown(f"*{emotion_expl}*")
        st_lottie(lottie_type, height=200)

        # Word Frequency & Word Cloud
    step += 1
    progress.progress(min(step / steps, 1.0))
    with st.expander("📊 Word Analysis"):
        st.markdown("<div class='expander-description'>Visualize word frequency and emotional color palette</div>", unsafe_allow_html=True)
        # Word Cloud
        wc, wc_expl = generate_wordcloud(poem, dominant_emotion)
        if wc:
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt.gcf())
            st.caption(wc_expl)
        else:
            st.warning(f"Could not generate word cloud: {wc_expl}")

        # Word Frequency
        top_words, word_freq, word_expl = get_top_words(poem)
        if top_words:
            df = pd.DataFrame(top_words, columns=["Word", "Frequency"])
            fig = px.bar(df, x="Word", y="Frequency")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(word_expl)
        else:
            st.warning(f"Could not analyze word frequency: {word_expl}")

        # Color Palette
        palette, palette_expl = generate_color_palette(EMOTION_KEYWORDS[dominant_emotion]["color"])
        st.markdown("**Emotional Color Palette**")
        cols = st.columns(len(palette))
        for i, color in enumerate(palette):
            with cols[i]:
                st.markdown(
                    f"<div style='background-color:{color};width:40px;height:40px;border-radius:5px;border:1px solid #ffffff;'></div>",
                    unsafe_allow_html=True
                )
                st.markdown(f"<small>{color}</small>", unsafe_allow_html=True)
        st.markdown(f"*{palette_expl}*")
    # Figurative Language
    step += 1
    progress.progress(min(step / steps, 1.0))
    figures, fig_expl = detect_figurative_language(poem)
    with st.expander("🎭 Literary Devices"):
        st.markdown("<div class='expander-description'>Discover the rhetorical devices that enrich your poem</div>", unsafe_allow_html=True)
        if figures:
            for fig_type, line, conf, expl in figures[:10]:  # Show top 10
                st.markdown(f"- **{fig_type}** (Confidence: {conf:.2f}): *{line.strip()}*")
                st.markdown(f"  <small>{expl}</small>", unsafe_allow_html=True)
            st.markdown(f"*{fig_expl}*")
        else:
            st.markdown("No significant literary devices detected")

    st_lottie(lottie_search, height=200)

    # Structure
    step += 1
    progress.progress(min(step / steps, 1.0))
    total_syllables, syll_expl = count_syllables_total(poem)
    poem_length, len_expl = categorize_poem_length(total_words)
    with st.expander("📐 Poem Structure"):
        st.markdown("<div class='expander-description'>Analyze the formal structure of your poem</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Lines", len(poem.strip().splitlines()))
        col2.metric("Stanzas", len([s for s in poem.split('\n\n') if s.strip()]) or 1)
        col3.metric("Syllables", total_syllables)
        st.markdown(f"**Length**: {poem_length}")
        st.markdown(f"*{len_expl}*")
        st.markdown(f"*{syll_expl}*")

       
        # Themes
    step += 1
    progress.progress(min(step / steps, 1.0))
    theme_scores, theme_expl = analyze_themes(poem)
    themes = [theme for theme, score in sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)[:3]]
    with st.expander("🌟 Thematic Insights"):
        st.markdown("<div class='expander-description'>Identify the central themes in your poem</div>", unsafe_allow_html=True)
        st.markdown(f"**Dominant Themes**: {', '.join(themes[:3])}")
        theme_df = pd.DataFrame.from_dict(theme_scores, orient='index', columns=['Score'])
        fig_theme = px.bar(
            theme_df, x=theme_df.index, y='Score', title="Theme Intensity",
            color=theme_df.index, color_discrete_sequence=px.colors.qualitative.Pastel,
            text=theme_df['Score'].apply(lambda x: f"{x:.2f}")
        )
        fig_theme.update_layout(
            plot_bgcolor='#1a1a1a', paper_bgcolor='#1a1a1a', font_color='#FFFFFF',
            xaxis_title="Themes", yaxis_title="Intensity", xaxis_tickangle=45,
            bargap=0.10, 
        )
        st.plotly_chart(fig_theme, use_container_width=False)
        st.markdown(f"*{theme_expl}*")
    st_lottie(lottie_book, height=200)
    # Poetic Identity
    step += 1
    progress.progress(min(step / steps, 1.0))
    badge, badge_expl = generate_poetic_badge(diversity, figures, themes, theme_scores, emotion_scores)
    archetype, arc_expl = classify_narrative_archetype(poem, themes, theme_scores, emotion_scores, sentiment)
    with st.expander("🏅 Poetic Identity"):
        st.markdown("<div class='expander-description'>Discover your unique poetic style and narrative patterns</div>", unsafe_allow_html=True)
        st.markdown(f"**Poetic Badge**: {badge}")
        st.markdown(f"*{badge_expl}*")
        st.markdown(f"**Narrative Archetype**: {archetype}")
        st.markdown(f"*{arc_expl}*")
    st_lottie(lottie_grammy, height=350)

        # Playlist
    step += 1
    progress.progress(min(step / steps, 1.0))
    songs_db = load_song_data()
    if not songs_db:
        st.warning("Could not load song database. Playlist generation disabled.")
    else:
        playlist, playlist_expl = generate_playlist(emotion_scores, songs_db, top_n=5)
        with st.expander("🎧 Playlist Generator"):
            st.markdown("<div class='expander-description'>Get song recommendations that match your poem's emotional tone</div>", unsafe_allow_html=True)
            if playlist:
                st.markdown(f"*{playlist_expl}*")
                for i, song in enumerate(playlist, 1):
                    st.markdown(f"**{i}. {song.get('title', 'Unknown Title')}** by *{song.get('artist', 'Unknown Artist')}*")
                    
                    # Display album art if available
                    if song.get('image_url'):
                        st.image(song['image_url'], width=100)
            
            # Handle preview URL
            preview_url = song.get('preview_url')
            if preview_url and preview_url.lower() not in ['none', 'null', '']:
                st.audio(preview_url, format='audio/mp3')
            
            # Show lyrics snippet
                lyrics = song.get('lyrics', 'No lyrics available')
                st.markdown(f"<small>{lyrics[:150]}...</small>", unsafe_allow_html=True)
                st.markdown("---")
    # Reports and Summary
    full_report = generate_analysis_report(
        poem, sentiment, subjectivity, mood, reading_score, reading_level,
        emotion_scores, total_syllables, poem_length, themes, theme_scores, figures,
        badge, archetype, themes
    )
    
    latex_report = generate_latex_report(
        poem, sentiment, subjectivity, mood, reading_score, reading_level,
        emotion_scores, total_syllables, poem_length, themes, theme_scores, figures,
        badge, archetype
    )
    
    summary = generate_summary(
        poem, sentiment, mood, themes, emotion_scores, badge, archetype
    )

    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📄 Download Text Report",
            data=full_report,
            file_name=f"poem_analysis_{st.session_state.analysis_id}.txt",
            mime="text/plain"
        )
    with col2:
        st.download_button(
            label="📄 Download LaTeX Report",
            data=latex_report,
            file_name=f"poem_analysis_{st.session_state.analysis_id}.tex",
            mime="text/x-tex"
        )

    # Summary display
    st.markdown("### 📋 Analysis Summary")
    st.text_area(
        "Copy this summary to share:",
        summary,
        height=200,
        key="summary_text"
    )

    # Analysis Complete
    st.balloons()
    st.success("Analysis complete! Explore the tabs above for detailed insights.")
    logger.info(f"Analysis completed for session {st.session_state.analysis_id}")
