# SGLang 0.5.19 on cu130 already carries the exact pins Ricursive's DGX Spark recipe asks for
# (torch 2.13.0+cu130, transformers 5.12.1, flashinfer 0.6.18). Ripple's model class, kernels, tree
# adapter and chat parsers are out-of-tree: everything except serve.py is vendored verbatim from
# ricursiveAI/Ripple-65B-Preview-NVFP4 (inference/local/ at 2789217b, Apache-2.0; chat_template.jinja
# from the repo root). serve.py is our HTTP wrapper around their spark.py.
FROM lmsysorg/sglang:v0.5.19-cu130
COPY *.py chat_template.jinja /opt/ripple/
WORKDIR /opt/ripple
ENTRYPOINT ["python3", "/opt/ripple/serve.py"]
