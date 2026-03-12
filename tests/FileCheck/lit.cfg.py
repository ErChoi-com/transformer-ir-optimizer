import os

import lit.formats

config.name = "LLMInferencePass"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = [".ll"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.test_source_root

config.substitutions.append(("%opt", getattr(config, "opt", "opt")))
config.substitutions.append(("%FileCheck", getattr(config, "FileCheck", "FileCheck")))
config.substitutions.append(("%shlibext", getattr(config, "shlibext", ".dll")))
config.substitutions.append(("%shlibdir", getattr(config, "shlibdir", os.path.join(config.test_source_root, "..", "..", "build"))))