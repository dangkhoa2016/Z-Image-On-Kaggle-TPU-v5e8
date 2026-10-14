import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn

class ChunkedZSingleStreamAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self, chunk_size: int, sync_chunks: bool = False):
        self.chunk_size = int(chunk_size)
        self.sync_chunks = bool(sync_chunks)
        if self.chunk_size <= 0:
            raise ValueError('chunk_size must be > 0')

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, freqs_cis=None):
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        def apply_rotary_emb(x_in, freqs):
            with torch.amp.autocast('cuda', enabled=False):
                x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
                f = freqs.unsqueeze(2)
                return torch.view_as_real(x * f).flatten(3).type_as(x_in)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        q_len = query.shape[1]
        if q_len <= self.chunk_size:
            out = dispatch_attention_fn(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0,
                is_causal=False, backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
        else:
            pieces = []
            for start in range(0, q_len, self.chunk_size):
                end = min(start + self.chunk_size, q_len)
                piece = dispatch_attention_fn(
                    query[:, start:end], key, value,
                    attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                    backend=self._attention_backend, parallel_config=self._parallel_config,
                )
                if self.sync_chunks:
                    import torch_xla
                    torch_xla.sync()
                pieces.append(piece)
            out = torch.cat(pieces, dim=1)

        out = out.flatten(2, 3).to(dtype)
        out = attn.to_out[0](out)
        if len(attn.to_out) > 1:
            out = attn.to_out[1](out)
        return out
