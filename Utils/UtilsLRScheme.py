def deeplab_scheme_builder(batches:int, epochs:int, warmup=True, warmup_epochs=1, warmup_factor=1e-3):

    assert batches > 0 and epochs > 0, "参数范围错误"
    if(not warmup): warmup_epochs = 0

    def deeplab_scheme(batch_count):
        batch_all = warmup_epochs * batches
        return warmup_factor * (1 - float(batch_count) / batch_all) + float(batch_count) / batch_all \
            if(warmup and batch_count <= batch_all) else (1 - (batch_count - batch_all) / (epochs * batches - batch_all)) ** 0.9
    
    return deeplab_scheme

def self_scheme_builder(batches:int, hun_to=0.3, warmup=True, warmup_epochs=1, warmup_factor=1e-3):

    assert batches > 0, "参数范围错误"
    if(not warmup): warmup_epochs = 0

    def self_scheme(batch_count):
        batch_all = warmup_epochs * batches
        return warmup_factor * (1 - float(batch_count) / batch_all) + float(batch_count) / batch_all \
             if(warmup and batch_count <= batch_all) else (100 - 100 * hun_to) / (99 * batch_count) + (100 * hun_to - 1) / 99
    
    return self_scheme