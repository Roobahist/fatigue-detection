
def get_embedding(
    embedding_model_name: str,
    model_bundle: Tuple,
    frame: np.ndarray,
    strategies: Union[str, List[str]] = "cls",
) -> Dict[str, np.ndarray]:
    """
    Get one or more embeddings for a single frame using specified strategies.

    Args:
        embedding_model_name (str): Model name for validation.
        model_bundle (Tuple): (processor, model) from get_embedding_model().
        frame (np.ndarray): The input frame in HWC RGB format.
        strategies (Union[str, List[str]]): Strategy or list of strategies.
            Options: "cls", "cls+mean", "pooled", "tokens", "tokens+cls"

    Returns:
        Dict[str, np.ndarray]: A dict mapping strategy names to their embeddings.
    """
    # todo : make it compatible with Facenet and other models

    if "/" not in embedding_model_name:
        raise ValueError(f"Unsupported model: {embedding_model_name}")

    if isinstance(strategies, str):
        strategies = [strategies]

    processor, model = model_bundle
    image = Image.fromarray(frame)
    inputs = processor(images=image, return_tensors="pt")

    result = {}

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state  # [1, num_tokens, hidden_dim]
        pooler_output = outputs.pooler_output  # [1, hidden_dim] or None

        cls_token = last_hidden_state[:, 0, :]  # [1, hidden_dim]
        patch_tokens = last_hidden_state[:, 1:, :]  # [1, 196, hidden_dim]
        mean_pooled = patch_tokens.mean(dim=1)  # [1, hidden_dim]
        cls_plus_mean = torch.cat([cls_token, mean_pooled], dim=1)  # [1, 2*hidden_dim]

        for strategy in strategies:
            if strategy == "cls":
                result["cls"] = cls_token.squeeze(0).cpu().numpy().astype(np.float32)

            elif strategy == "cls+mean":
                result["cls+mean"] = (
                    cls_plus_mean.squeeze(0).cpu().numpy().astype(np.float32)
                )

            elif strategy == "pooled":
                if pooler_output is not None:
                    result["pooled"] = (
                        pooler_output.squeeze(0).cpu().numpy().astype(np.float32)
                    )
                else:
                    raise ValueError(
                        "This model does not support pooled output (pooler_output is None)."
                    )

            elif strategy == "tokens":
                flat_tokens = (
                    patch_tokens.squeeze(0).flatten().cpu().numpy().astype(np.float32)
                )  # shape: [196 * hidden_dim]
                result["tokens"] = flat_tokens

            elif strategy == "tokens+cls":
                full_tokens = (
                    last_hidden_state.squeeze(0)
                    .flatten()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )  # shape: [197 * hidden_dim]
                result["tokens+cls"] = full_tokens

            else:
                raise ValueError(f"Unsupported strategy: {strategy}")

    return result