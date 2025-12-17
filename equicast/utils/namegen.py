"""Cute name generator for experiments."""

import random


# Session-wide cute name - generated once per Python session
_SESSION_NAME = None


def cute() -> str:
    """
    Get the cute name for this Python session.

    The name is generated once when first called and remains consistent
    throughout the session. Useful for naming experiments, runs, and logs
    consistently across all components.

    Returns:
        The cute name for this session (e.g., "happy-panda")

    Example:
        >>> from equicast import cute
        >>> name1 = cute()
        >>> name2 = cute()
        >>> assert name1 == name2  # Same name throughout session
    """
    global _SESSION_NAME
    if _SESSION_NAME is None:
        _SESSION_NAME = generate_name()
    return _SESSION_NAME


def reset_cute() -> None:
    """
    Reset the session cute name. Mainly useful for testing.

    After calling this, the next call to cute() will generate a new name.
    """
    global _SESSION_NAME
    _SESSION_NAME = None


ADJECTIVES = [
    "happy", "jolly", "brave", "clever", "bright", "swift", "gentle", "calm",
    "wise", "quiet", "loud", "tiny", "grand", "proud", "shy", "bold",
    "keen", "warm", "cool", "fresh", "wild", "tame", "free", "kind",
    "noble", "fair", "sweet", "spicy", "zesty", "merry", "chipper", "peppy",
    "snappy", "zippy", "bouncy", "fluffy", "fuzzy", "cozy", "snug", "dapper",
    "fancy", "glowing", "shiny", "sparkly", "misty", "cloudy", "sunny", "starry",
    # Additional adjectives for more combinations
    "agile", "amber", "ancient", "azure", "breezy", "bronze", "bustling", "charming",
    "crimson", "crystal", "dancing", "daring", "dashing", "divine", "dreamy", "eager",
    "elegant", "emerald", "epic", "fearless", "fierce", "flaming", "flying", "frosty",
    "gilded", "gleaming", "golden", "graceful", "hardy", "honest", "humble", "icy",
    "ivory", "jade", "jovial", "joyful", "lively", "lucky", "lunar", "lusty",
    "majestic", "mighty", "modest", "mystic", "nimble", "onyx", "pearl", "playful",
    "plucky", "radiant", "rapid", "regal", "robust", "rosy", "royal", "ruby",
    "rustic", "sacred", "scarlet", "serene", "shadow", "silver", "sleek", "smooth",
    "soaring", "solar", "solid", "sonic", "cosmic", "stellar", "sturdy", "subtle",
    "tender", "topaz", "tranquil", "trusty", "turbo", "vibrant", "vivid", "whimsy",
]

NOUNS = [
    "panda", "otter", "fox", "wolf", "bear", "deer", "owl", "hawk",
    "eagle", "falcon", "raven", "crow", "swan", "duck", "goose", "crane",
    "dolphin", "whale", "seal", "penguin", "turtle", "frog", "newt", "gecko",
    "tiger", "lion", "leopard", "cheetah", "lynx", "bobcat", "panther", "jaguar",
    "rabbit", "hare", "squirrel", "chipmunk", "beaver", "badger", "mole", "shrew",
    "mountain", "river", "lake", "ocean", "forest", "meadow", "valley", "peak",
    "cloud", "storm", "breeze", "gale", "wave", "tide", "current", "stream",
    # Additional nouns for more combinations
    "albatross", "alpaca", "antelope", "armadillo", "barracuda", "bison", "buffalo", "butterfly",
    "cardinal", "caribou", "catfish", "chameleon", "condor", "coyote", "cricket", "dragonfly",
    "elk", "ferret", "finch", "firefly", "flamingo", "gazelle", "giraffe", "groundhog",
    "hamster", "hedgehog", "heron", "hornet", "hummingbird", "iguana", "impala", "jackal",
    "kangaroo", "kingfisher", "koala", "lemur", "llama", "lobster", "macaw", "mallard",
    "manta", "marlin", "meerkat", "mongoose", "moose", "narwhal", "nautilus", "nightingale",
    "ocelot", "octopus", "oriole", "osprey", "ostrich", "pelican", "phoenix", "puffin",
    "quail", "raccoon", "raptor", "rattlesnake", "reindeer", "rhino", "salamander", "salmon",
    "scorpion", "seagull", "seahorse", "shark", "sparrow", "starfish", "stingray", "swallow",
    "swordfish", "toucan", "trout", "vulture", "walrus", "warthog", "weasel", "woodpecker",
    "zebra", "canyon", "cliff", "comet", "crater", "desert", "dune", "fjord",
    "glacier", "harbor", "horizon", "island", "lagoon", "meteor", "nebula", "oasis",
    "planet", "prairie", "quasar", "reef", "ridge", "savanna", "summit", "tundra",
    "vortex", "waterfall", "willow", "zenith",
]


def generate_name(separator: str = "-") -> str:
    """
    Generate a cute random name.

    Args:
        separator: String to use between adjective and noun

    Returns:
        A randomly generated cute name like "happy-panda"

    Examples:
        >>> name = generate_name()
        >>> assert isinstance(name, str)
        >>> assert "-" in name
        >>> parts = name.split("-")
        >>> assert len(parts) == 2
    """
    adjective = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)
    return f"{adjective}{separator}{noun}"


def generate_unique_names(count: int, separator: str = "-") -> list[str]:
    """
    Generate a list of unique cute names.

    Args:
        count: Number of unique names to generate
        separator: String to use between adjective and noun

    Returns:
        List of unique randomly generated names

    Raises:
        ValueError: If count exceeds possible unique combinations
    """
    max_combinations = len(ADJECTIVES) * len(NOUNS)
    if count > max_combinations:
        raise ValueError(
            f"Cannot generate {count} unique names. "
            f"Maximum possible combinations: {max_combinations}"
        )

    names = set()
    while len(names) < count:
        names.add(generate_name(separator))

    return list(names)
