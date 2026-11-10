"""Real client-server federated infrastructure: a server process (with a
dashboard) and separate client processes (each its own local "room") that
train a node locally and push results over HTTP, driving the sync/async
round protocol from the exact same config files fedyolo.simulate uses
in-process. See rooms.py, server_app.py, client_app.py.
"""
