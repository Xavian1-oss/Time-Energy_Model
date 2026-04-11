import os

from jlclient import jarvisclient
from jlclient.jarvisclient import *

jarvisclient.token = "YoCDBRncnI85PcGc3UJ2rP00D5jkS8G6vc6NzJaZ_sA"


def pause():
    machine_id = os.getenv("MACHINE_ID")
    instance = User.get_instance(machine_id)
    instance.pause()


pause()
