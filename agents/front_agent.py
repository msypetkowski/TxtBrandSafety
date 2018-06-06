from pulsar.apps.wsgi import Router
from pulsar.apps.wsgi.handlers import WsgiHandler
from pulsar.apps.wsgi import WSGIServer, WsgiResponse


blueprint = Router('/')


@blueprint.router('compat', methods=['get', 'post'])
def calc_compatibility(request):
    data = b'[<there will be numbers>]'
    return WsgiResponse(200, data)


# @blueprint.router('async', methods=['delete', 'put'])
# async def async_cast(request):
#     return WsgiResponse(200, 'async')


def server(**kwargs):
    return WSGIServer(callable=WsgiHandler((blueprint, )), **kwargs)


if __name__ == '__main__':
    server().start()
