import unittest
from types import SimpleNamespace
from unittest.mock import patch
from tools.pi05_feedback_ray_isolation import install


class RayIsolationTest(unittest.TestCase):
    def test_auto_is_local_without_losing_rlinf_runtime_options(self):
        calls=[]
        fake=SimpleNamespace(init=lambda *a,**kw:calls.append((a,kw)))
        with patch.dict('sys.modules',{'ray':fake}):
            install();fake.init(address='auto',namespace='RLinf',runtime_env={'env_vars':{'A':'b'}})
        args,kwargs=calls[0]
        self.assertEqual(kwargs['address'],'local')
        self.assertEqual(kwargs['namespace'],'RLinf')
        self.assertEqual(kwargs['runtime_env'],{'env_vars':{'A':'b'}})
        self.assertEqual(kwargs['object_store_memory'],16*1024**3)

    def test_positional_address_and_explicit_memory_cap(self):
        calls=[]
        fake=SimpleNamespace(init=lambda *a,**kw:calls.append((a,kw)))
        with patch.dict('sys.modules',{'ray':fake}):
            install();fake.init('auto',object_store_memory=8*1024**3)
        self.assertEqual(calls[0][0],('local',))
        self.assertEqual(calls[0][1]['object_store_memory'],8*1024**3)


if __name__=='__main__':unittest.main()
