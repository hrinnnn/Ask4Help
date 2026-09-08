"""Task-local Ray startup policy; never connect this ablation to another job."""
def install():
    import ray
    original=ray.init
    def initialize(*args,**kwargs):
        # Pinned RLinf asks for address=auto and ignores RLINF_RAY_ADDRESS.
        # Keep its namespace/runtime_env, but start a separate local cluster.
        if args:
            args=('local',*args[1:]);kwargs.pop('address',None)
        else:kwargs['address']='local'
        kwargs.setdefault('object_store_memory',16*1024**3)
        return original(*args,**kwargs)
    ray.init=initialize
