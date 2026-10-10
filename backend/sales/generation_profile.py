"""Per-call generation observations; never capture prompts or output contents."""
import time


def generation_kwargs(max_new_tokens, eos_token_id, stopping_criteria):
    return dict(max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=eos_token_id, stopping_criteria=stopping_criteria)


class TokenTimer:
    # Transformers sends the prompt first, then generated token tensors.
    def __init__(self, audit):
        self.audit = audit
        self.prompt_seen = False
        self.first = None
        self.last = None
        self.count = 0
        self.start = time.monotonic()

    def put(self, value):
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        # Synchronize only the first token so TTFT includes CUDA execution.
        if self.first is None:
            try:
                import torch
                if value.is_cuda:
                    torch.cuda.synchronize(value.device)
            except (ImportError, AttributeError):
                pass
        now = time.monotonic()
        self.first = now if self.first is None else self.first
        self.last = now
        self.count += int(value.numel())

    def end(self):
        decode = max(0, (self.last or self.start) - (self.first or self.start))
        self.audit.update(time_to_first_token_seconds=None if self.first is None else round(self.first-self.start, 4),
                          decode_seconds=round(decode, 4), output_tokens=self.count,
                          decode_tokens_per_second=round((self.count-1)/decode, 3) if decode and self.count > 1 else None)

    def start_new_attempt(self):
        self.end()
        fields=('time_to_first_token_seconds','decode_seconds','output_tokens','decode_tokens_per_second')
        self.audit.setdefault('prior_protocol_attempt_profiles',[]).append({k:self.audit.get(k) for k in fields})
        self.audit['profile_scope']='last_protocol_attempt'
        self.prompt_seen=False; self.first=None; self.last=None; self.count=0; self.start=time.monotonic()


class AttemptTimer(TokenTimer):
    """Observe each protocol attempt without replacing caller streaming."""
    def __init__(self, audit, delegate=None):
        super().__init__(audit)
        self.delegate = delegate

    def put(self, value):
        super().put(value)
        if self.delegate is not None:
            self.delegate.put(value)

    def end(self):
        super().end()
        self.audit['token_stream_observed'] = self.first is not None
        self.audit['timing_boundary'] = 'client_generate_call_including_transport_not_pure_gpu_prefill'
        if self.delegate is not None:
            self.delegate.end()


def profiled_generate(model, inputs, *, audit, contract=None, **kwargs):
    import torch
    audit['input_tokens'] = int(inputs['input_ids'].shape[-1])
    audit['device_map'] = {str(k): str(v) for k, v in getattr(model, 'hf_device_map', {}).items()}
    audit['model_device'] = str(getattr(model,'device','unknown'))
    config = getattr(model, 'config', None)
    audit['attention_implementation'] = {
        name: getattr(getattr(config, name, config), '_attn_implementation', None)
        for name in ('text_config', 'vision_config')}
    # Some Transformers versions do not publish hf_device_map. Do not turn
    # missing placement observations into an assertion of no CPU offload.
    parameter_devices = {}
    if hasattr(model, 'parameters'):
        for parameter in model.parameters():
            device = str(parameter.device)
            parameter_devices[device] = parameter_devices.get(device, 0) + parameter.numel()
    audit['parameter_device_counts'] = parameter_devices
    placements = list(audit['device_map'].values()) + list(parameter_devices)
    audit['cpu_or_disk_offload'] = any(v in {'cpu','disk','meta'} for v in placements) if placements else None
    remote=getattr(model,'is_remote_model',False)
    if remote:
        audit['input_transport_device']='cpu'
        audit['model_device']='independent_model_service'
    if torch.cuda.is_available() and not remote:
        torch.cuda.reset_peak_memory_stats()
        audit.update(gpu_allocated_before=torch.cuda.memory_allocated(), gpu_reserved_before=torch.cuda.memory_reserved())
        free,total=torch.cuda.mem_get_info()
        audit.update(gpu_free_before=free,gpu_total_bytes=total)
    try:
        import psutil
        audit['cpu_rss_bytes'] = psutil.Process().memory_info().rss
    except ImportError:
        pass
    timer = TokenTimer(audit)
    started = time.monotonic()
    try:
        from backend.sales.model_contract import checked_generate
        output = checked_generate(model, contract=contract, **inputs, streamer=timer, **kwargs)
        audit['output_tokens'] = int(output.shape[-1] - inputs['input_ids'].shape[-1])
        eos = getattr(getattr(model, 'generation_config', None), 'eos_token_id', [])
        eos = eos if isinstance(eos, list) else [eos]
        last_token = int(output[0, -1])
        audit['stop_reason'] = ('output_limit' if audit['output_tokens'] >= kwargs['max_new_tokens']
                                else 'eos' if last_token in eos else 'stopping_criteria')
        return output
    except Exception as exc:
        audit['error_type'] = type(exc).__name__
        raise
    finally:
        timer.end()
        audit['generation_seconds'] = round(time.monotonic()-started, 4)
        if remote:
            audit.update(getattr(model,'last_runtime_profile',{}))
            audit['gpu_metrics_origin']='independent_model_service'
        if torch.cuda.is_available() and not remote:
            audit.update(gpu_peak_allocated=torch.cuda.max_memory_allocated(),
                         gpu_allocated_after=torch.cuda.memory_allocated(), gpu_reserved_after=torch.cuda.memory_reserved())
            audit['gpu_free_after']=torch.cuda.mem_get_info()[0]
