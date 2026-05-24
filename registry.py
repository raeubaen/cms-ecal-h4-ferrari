RECO_REGISTRY = {}

def get_reco(name):
    return RECO_REGISTRY[name]

def register_reco(name):
    def wrapper(fn):
        RECO_REGISTRY[name] = fn
        return fn
    return wrapper
