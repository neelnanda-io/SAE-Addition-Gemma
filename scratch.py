# %%
from neel.imports import *
import transformer_lens
from transformer_lens import (
    HookedTransformerConfig,
    HookedTransformer,
    FactoredMatrix,
    ActivationCache,
)
import sae_lens
from sae_lens import HookedSAETransformer
from neel_plotly import *

# %%
torch.set_grad_enabled(False)
# %%
model = HookedSAETransformer.from_pretrained("gemma-2-2b")
d_model = model.cfg.d_model
d_head = model.cfg.d_head
n_layers = model.cfg.n_layers
print(model.cfg)
# %%
numbers = [1, 2, 34, 192, 964, 832, 18634, 18436621245387]
for n in numbers:
    print(model.to_str_tokens(str(n)))
# %%
model.generate("294 + 539 = 833\n410 + 161 = 571")


# %%
def generate_addition_dataset(n=1000, upper_bound=499, lower_bound=100):
    outputs = []
    for _ in range(n):
        temp = []
        for _ in range(2):
            a = random.randint(lower_bound, upper_bound)
            b = random.randint(lower_bound, upper_bound)
            temp.append(f"{a}+{b}={a+b}")
        outputs.append("\n".join(temp))
    return outputs


dataset = generate_addition_dataset(100)
# %%
number_tokens = model.to_tokens(list(map(str, range(10))))[:, 1]
print(number_tokens)
tokens = model.to_tokens(dataset)
# %%
logits = model(tokens)[:, -4:-1, :]
print(logits.shape)
# %%
probs = F.softmax(logits, dim=-1)
number_probs = probs.gather(dim=-1, index=tokens[:, -3:, None]).squeeze(-1)
number_probs.shape
imshow(number_probs, y=[d for d in dataset])
# %%
# for i in range(10):
# _ = model(tokens, return_type="loss")
clean_dataset = generate_addition_dataset(200)
corr_dataset = generate_addition_dataset(200)
corr_dataset = [cl[:12] + cr[12:] for cl, cr in zip(clean_dataset, corr_dataset)]
print(corr_dataset[0])
print(clean_dataset[0])

clean_tokens = model.to_tokens(clean_dataset)[:, :-2]
corr_tokens = model.to_tokens(corr_dataset)[:, :-2]
are_hundreds_equal = clean_tokens[:, -1] == corr_tokens[:, -1]
clean_tokens = clean_tokens[~are_hundreds_equal][:100]
corr_tokens = corr_tokens[~are_hundreds_equal][:100]
print(clean_tokens.shape)
print(corr_tokens.shape)


# %%
def hundreds_loss(logits, tokens=clean_tokens):
    if len(logits.shape) == 3:
        logits = logits[:, -2, :]
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs[torch.arange(log_probs.shape[0]), tokens[:, -1]].mean()


# %%

clean_logits, clean_cache = model.run_with_cache(
    clean_tokens, names_filter=lambda name: "resid_post" in name
)
clean_loss = hundreds_loss(clean_logits, clean_tokens)

corr_logits, corr_cache = model.run_with_cache(
    corr_tokens, names_filter=lambda name: "resid_post" in name
)
corr_loss = hundreds_loss(corr_logits, clean_tokens)
print(f"{corr_loss.item()=}")
print(f"{clean_loss.item()=}")


# %%


def activation_patch(
    layer: int,
    pos: int,
    tokens: torch.Tensor,
    cache: ActivationCache,
) -> torch.Tensor:
    """
    Perform activation patching between two tensors of tokens for the residual stream at a given token and layer.

    Args:
    model: The transformer model
    clean_tokens: The tokens for the clean (correct) input
    corrupted_tokens: The tokens for the corrupted input
    layer: The layer to patch at
    pos: The position to patch at

    Returns:
    The logits after patching
    """

    def patching_hook(resid, hook):
        resid[:, pos, :] = cache["resid_post", layer][:, pos, :]
        return resid

    # Run the model on corrupted input with the patching hook
    logits = model.run_with_hooks(
        tokens, fwd_hooks=[(f"blocks.{layer}.hook_resid_post", patching_hook)]
    )

    return hundreds_loss(logits)


results = []
POS_LABELS = ["H1", "H2", "="]
for layer in tqdm.trange(0, n_layers, 5):
    for c, pos in enumerate([13, 17, 20]):
        noised_loss = activation_patch(layer, pos, clean_tokens, corr_cache)
        denoised_loss = activation_patch(layer, pos, corr_tokens, clean_cache)
        results.append(
            {
                "label": POS_LABELS[c],
                "denoised": denoised_loss.item(),
                "noised": noised_loss.item(),
                "layer": layer,
                "pos": pos,
            }
        )
patching_df = pd.DataFrame(results)
px.imshow(
    patching_df,
    x="label",
    y="layer",
    z="denoised",
    color_continuous_midpoint=0,
    color_continuous_scale="RdBu",
)


# %%
def normalize_loss(loss):
    return (loss - corr_loss.item()) / (clean_loss.item() - corr_loss.item())


imshow(
    normalize_loss(
        np.array(
            [
                [
                    patching_df["denoised"][
                        (patching_df["layer"] == layer) & (patching_df["label"] == pos)
                    ].item()
                    for pos in POS_LABELS
                ]
                for layer in range(0, n_layers, 5)
            ]
        )
    ),
    x=POS_LABELS,
    y=list(range(0, n_layers, 5)),
    aspect="auto",
    color_continuous_midpoint=0,
    color_continuous_scale="RdBu",
    title="Denoised Loss",
)
imshow(
    normalize_loss(
        np.array(
            [
                [
                    patching_df["noised"][
                        (patching_df["layer"] == layer) & (patching_df["label"] == pos)
                    ].item()
                    for pos in POS_LABELS
                ]
                for layer in range(0, n_layers, 5)
            ]
        )
    ),
    x=POS_LABELS,
    y=list(range(0, n_layers, 5)),
    aspect="auto",
    color_continuous_midpoint=0,
    color_continuous_scale="RdBu",
    title="Noised Loss",
)


# %%
results = []
POS_LABELS = ["H1", "H2", "="]
layers = list(range(10, 20))

for layer in tqdm.tqdm(layers):
    for c, pos in enumerate([13, 17, 20]):
        noised_loss = activation_patch(layer, pos, clean_tokens, corr_cache)
        denoised_loss = activation_patch(layer, pos, corr_tokens, clean_cache)
        results.append(
            {
                "label": POS_LABELS[c],
                "denoised": denoised_loss.item(),
                "noised": noised_loss.item(),
                "layer": layer,
                "pos": pos,
            }
        )
patching_df = pd.DataFrame(results)
px.imshow(
    patching_df,
    x="label",
    y="layer",
    color_continuous_midpoint=0,
    color_continuous_scale="RdBu",
)

# %%
px.line(
    patching_df, color="label", x="layer", y="denoised", title="Denoised Loss"
).show()
px.line(patching_df, color="label", x="layer", y="noised", title="Noised Loss").show()

# %%
import yaml

with open("/workspace/SAELens/sae_lens/pretrained_saes.yaml", "r") as file:
    pretrained_saes = yaml.safe_load(file)
print(pretrained_saes.keys())
# %%
RELEASE = "gemma-scope-2b-pt-res-canonical"
width = 65
layer = 18

sae_18 = sae_lens.SAE.from_pretrained(
    release=RELEASE,
    sae_id=f"layer_{layer}/width_{width}k/canonical",
    device="cuda",
)

# Attach the SAE model to your existing HookedSAETransformer
model.add_sae(sae_18, f"blocks.{layer}.hook_resid_post")
# %%
layer = 24
sae_24 = sae_lens.SAE.from_pretrained(
    release=RELEASE,
    sae_id=f"layer_{layer}/width_{width}k/canonical",
    device="cuda",
)[0]


# Attach the SAE model to your existing HookedSAETransformer
model.add_sae(sae_24, f"blocks.{layer}.hook_resid_post")
# %%
big_dataset = generate_addition_dataset(1000)
big_tokens = model.to_tokens(big_dataset)[:, :-2]
# %%
model.reset_saes()
# filter_not_qkv_input = lambda name: "_input" not in name


def get_cache_fwd_and_bwd(model, tokens, metric, layers):
    model.reset_hooks()
    cache = {}

    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()

    for layer in layers:
        model.add_hook(f"blocks.{layer}.hook_resid_post", forward_cache_hook, "fwd")

    grad_cache = {}

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()

    for layer in layers:
        model.add_hook(f"blocks.{layer}.hook_resid_post", backward_cache_hook, "bwd")
    torch.set_grad_enabled(True)
    value = metric(model(tokens), tokens) * len(tokens)
    value.backward()
    torch.set_grad_enabled(False)
    model.reset_hooks()
    return (
        value.item(),
        ActivationCache(cache, model),
        ActivationCache(grad_cache, model),
    )


caches = []
grad_caches = []
losses = []

for i in tqdm.trange(0, len(big_tokens), 20):
    loss, cache, grad_cache = get_cache_fwd_and_bwd(
        model, big_tokens[i : i + 20], hundreds_loss, [18, 24]
    )
    losses.append(loss)
    caches.append(cache)
    grad_caches.append(grad_cache)
full_cache = {k: torch.cat([cache[k] for cache in caches]) for k in caches[0].keys()}
full_cache = ActivationCache(full_cache, model)
full_grad_cache = {
    k: torch.cat([grad_cache[k] for grad_cache in grad_caches])
    for k in grad_caches[0].keys()
}
full_grad_cache = ActivationCache(full_grad_cache, model)
# %%

# model: HookedSAETransformer
# model.reset_saes()


# def hundreds_loss(logits, tokens):
#     if len(logits.shape) == 3:
#         logits = logits[:, -2, :]
#     log_probs = F.log_softmax(logits, dim=-1)
#     return log_probs[torch.arange(log_probs.shape[0]), tokens[:, -1]].mean()

# torch.set_grad_enabled(True)

resid_18 = full_cache["blocks.18.hook_resid_post"][:, -2, :]
resid_24 = full_cache["blocks.24.hook_resid_post"][:, -2, :]
grad_18 = full_grad_cache["blocks.18.hook_resid_post"][:, -2, :]
grad_24 = full_grad_cache["blocks.24.hook_resid_post"][:, -2, :]

recons_resid_18, sae_cache_18 = sae_18.run_with_cache(resid_18)
sae_acts_18 = sae_cache_18["blocks.18.hook_resid_post.hook_sae_acts_post"]
recons_resid_24, sae_cache_24 = sae_24.run_with_cache(resid_24)
sae_acts_24 = sae_cache_24["blocks.24.hook_resid_post.hook_sae_acts_post"]


# %%
sae_attrs_18 = (grad_18 @ sae_18.W_dec.T) * sae_acts_18
sae_attrs_24 = (grad_24 @ sae_24.W_dec.T) * sae_acts_24
# %%
line([sae_attrs_18.mean(dim=0), sae_attrs_24.mean(dim=0)], title="SAE 18")
line([sae_attrs_18.std(dim=0), sae_attrs_24.std(dim=0)], title="Std SAE 18")
# %%
scatter(
    x=sae_attrs_18.mean(dim=0),
    y=sae_attrs_18.std(dim=0),
    xaxis="Mean SAE 18",
    yaxis="Std SAE 18",
)
scatter(
    x=sae_attrs_24.mean(dim=0),
    y=sae_attrs_24.std(dim=0),
    xaxis="Mean SAE 24",
    yaxis="Std SAE 24",
)
# %%
from IPython.display import IFrame


def get_dashboard_html(sae_release, sae_id, feature_idx):
    return f"https://neuronpedia.org/{sae_release}/{sae_id}/{feature_idx}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"


feature_idx = 54918
layer = 18

release, sae_id = sae_18.cfg.neuronpedia_id.split("/")
html = get_dashboard_html(sae_release=release, sae_id=sae_id, feature_idx=feature_idx)
display(IFrame(html, width=1200, height=600))

# %%
top_5_sae_18_latents = (-sae_attrs_18.mean(dim=0).abs()).argsort()[:5].tolist()
top_5_sae_24_latents = (-sae_attrs_24.mean(dim=0).abs()).argsort()[:5].tolist()
release, sae_id = sae_18.cfg.neuronpedia_id.split("/")
for f_id in top_5_sae_18_latents:
    html = get_dashboard_html(sae_release=release, sae_id=sae_id, feature_idx=f_id)
    display(IFrame(html, width=1200, height=600))
    scatter(
        x=sae_acts_18[:, f_id], y=sae_attrs_18[:, f_id], title=f"SAE 18 Feature {f_id}"
    )

release, sae_id = sae_24.cfg.neuronpedia_id.split("/")
for f_id in top_5_sae_24_latents:
    html = get_dashboard_html(sae_release=release, sae_id=sae_id, f_id=feature_idx)
    display(IFrame(html, width=1200, height=600))
    scatter(
        x=sae_acts_24[:, f_id], y=sae_attrs_24[:, f_id], title=f"SAE 24 Feature {f_id}"
    )

# %%
# for f_id in top_5_sae_18_latents:
# for f_id in top_5_sae_24_latents:

# %%
