__version__ = "0.1.0"


def __getattr__(name):
    if name == "ConceptDirectionExtractor":
        from safety_compass.concept import ConceptDirectionExtractor
        return ConceptDirectionExtractor
    if name == "register_strategy":
        from safety_compass.concept import register_strategy
        return register_strategy
    if name == "SafetyCompassMonitor":
        from safety_compass.monitor import SafetyCompassMonitor
        return SafetyCompassMonitor
    if name == "SafetyCompassCallback":
        from safety_compass.callback import SafetyCompassCallback
        return SafetyCompassCallback
    if name == "CompassCSVLogger":
        from safety_compass.logger import CompassCSVLogger
        return CompassCSVLogger
    if name == "SafetyCompassConfigError":
        from safety_compass.config import SafetyCompassConfigError
        return SafetyCompassConfigError
    if name == "load_experiment_config":
        from safety_compass.config import load_experiment_config
        return load_experiment_config
    if name == "get_formatter":
        from safety_compass.formatters import get_formatter
        return get_formatter
    if name == "register_formatter":
        from safety_compass.formatters import register_formatter
        return register_formatter
    if name == "default_behavioral_prompts":
        from safety_compass.behavioral import default_behavioral_prompts
        return default_behavioral_prompts
    if name == "evaluate_behavioral_prompts":
        from safety_compass.behavioral import evaluate_behavioral_prompts
        return evaluate_behavioral_prompts
    if name == "register_behavioral_scorer":
        from safety_compass.behavioral import register_behavioral_scorer
        return register_behavioral_scorer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
