import taurex
from taurex import chemistry

class ArrayGas(taurex.chemistry.Gas):
    """
    A Gas object for Tau-REx that allows the user to 
    pass in a custom abundance profile.
    """
    def __init__(self, molecule_name, abn):
        super().__init__('ArrayGas', molecule_name)
        self.abn = abn

    @property
    def mixProfile(self):
        return self.abn
