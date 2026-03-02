
_tracking_grad_enabled = True

class no_grad():
    def __enter__(self):
        global _tracking_grad_enabled
        self.prev = _tracking_grad_enabled
        _tracking_grad_enabled = False
    
    
    def __exit__(self, *args):
        global _tracking_grad_enabled
        _tracking_grad_enabled = self.prev
        return False

