# SGLang 0.5.19 on cu130 already carries the exact pins Ricursive's DGX Spark recipe asks for
# (torch 2.13.0+cu130, transformers 5.12.1, flashinfer 0.6.18). Ripple's model class and chat
# parsers are out-of-tree: model.py / ripple_format.py are vendored verbatim from
# ricursiveAI/Ripple-65B-Preview-NVFP4 (inference/local/, Apache-2.0), serve.py is our HTTP wrapper
# around their spark.py (which only offers an offline Engine).
FROM lmsysorg/sglang:v0.5.19-cu130
COPY model.py ripple_format.py serve.py /opt/ripple/
WORKDIR /opt/ripple
ENTRYPOINT ["python3", "/opt/ripple/serve.py"]
