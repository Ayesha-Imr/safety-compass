__version__ = "0.1.0"


def __getattr__(name):
    if name == "ConceptDirectionExtractor":
        from safety_compass.concept import ConceptDirectionExtractor
        return ConceptDirectionExtractor
    if name == "register_strategy":
        from safety_compass.concept import register_strategy
        return register_strategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
