# Sanity check: confirm the four required classes are defined before continuing.
_required = ["MultiFCPreprocessor", "TemporalSplitter", "DAMiniLMModel",
             "DAMiniLMExperimentRunner"]
_missing = [name for name in _required if name not in dir()]
assert not _missing, f"Missing class definitions: {_missing}."
print("All four project classes are defined:", ", ".join(_required))
