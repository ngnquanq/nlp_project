"""Local web serving support for the E1 translation model.

The package deliberately avoids importing FastAPI here so the shared Fairseq
compatibility helper remains usable in the original training environment.
"""
