"""Task-local Ray startup policy; never connect this ablation to another job."""
def install():
    import ray,os
    original=ray.init
    def initialize(*args,**kwargs):
        # Pinned RLinf asks for address=auto and ignores RLINF_RAY_ADDRESS.
        # Keep its namespace/runtime_env, but start a separate local cluster.
        if args:
            args=('local',*args[1:]);kwargs.pop('address',None)
        else:kwargs['address']='local'
        kwargs.setdefault('object_store_memory',16*1024**3)
        context=original(*args,**kwargs)
        # Ray state APIs otherwise rediscover every local cluster via "auto".
        os.environ['RAY_ADDRESS']=context.address_info['gcs_address']
        return context
    ray.init=initialize
