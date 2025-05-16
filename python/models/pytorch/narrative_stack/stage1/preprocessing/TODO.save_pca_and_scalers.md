# Save Parameters (instead of instances)

**Saving just the learned parameters** (instead of full `sklearn` objects) gives you:

* **Version-agnostic compatibility**
* **Cross-library portability** (e.g., if you want to reimplement in another framework)
* **Fewer surprises from internal changes**

---

### Example: Saving a `QuantileTransformer`

```python
scaler = QuantileTransformer(output_distribution='uniform')
scaler.fit(data)

params = {
    "quantiles_": scaler.quantiles_,
    "n_quantiles_": scaler.n_quantiles_,
    "references_": scaler.references_,
    "feature_names_in_": getattr(scaler, "feature_names_in_", None)
}

np.savez_compressed("scaler_params.npz", **params)
```

### Loading Later

```python
loaded = np.load("scaler_params.npz")
scaler = QuantileTransformer(output_distribution='uniform')
scaler.quantiles_ = loaded["quantiles_"]
scaler.n_quantiles_ = loaded["n_quantiles_"].item()
scaler.references_ = loaded["references_"]
if "feature_names_in_" in loaded:
    scaler.feature_names_in_ = loaded["feature_names_in_"]
```

Repeat similarly for `PCA`:

* Save `components_`, `mean_`, `explained_variance_`, etc.

---

This approach lets you **rebuild** the transformer manually with minimal coupling to `scikit-learn`’s object model.
