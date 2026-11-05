# SGLang with Clover support. Upstream sgl-project/sglang#46927 (Clover block-diffusion /
# self-speculative decoding) is still an unmerged PR, so no released image has it. The base
# nightly is dated just after the PR's base commit (2026-10-28), so its torch / transformers /
# sglang-kernel / flashinfer pins match the PR head; the PR is pure Python + Triton, so we
# install its tree over the image's sglang with --no-deps. Drop this once the PR is merged.
FROM lmsysorg/sglang:nightly-dev-cu13-20261031-a2cafa56

ARG CLOVER_REF=2243dc6edb416073b60be0765904cabae9eb61cf
RUN git clone --filter=blob:none https://github.com/sgl-project/sglang /opt/sglang-clover \
 && cd /opt/sglang-clover \
 && git fetch origin refs/pull/46927/head \
 && git checkout "$CLOVER_REF" \
 && pip install --no-cache-dir --no-deps -e python

# The PR's Triton kernels (BD-GDN conv, GDN, MoE coreset) are gated to compute capability 9/10,
# the only archs its author validated (H200). GB10 is SM12.1, so allow major 12 too. Unvalidated
# upstream; if Clover output looks wrong on this box, suspect these kernels first.
WORKDIR /opt/sglang-clover
RUN grep -rl 'get_device_capability(.*)\[0\] in (9, 10)' python/sglang/kernels/ops \
 | xargs sed -i 's/\[0\] in (9, 10)/[0] in (9, 10, 12)/' \
 && grep -rn 'in (9, 10, 12)' python/sglang/kernels/ops | wc -l
