#!/usr/bin/env python3
from flask import request, Flask

from agents.front_agent import FrontAgent

app = Flask(__name__)
agent = FrontAgent()


@app.route('/compat', methods=['GET', 'POST'])
def calc_compatibility():
    data = '\n'.join(list(request.form.keys()))
    data = data.encode()
    if 'data' in request.files:
        data = data + request.files['data'].read()
    print('-----got query of length:', len(data))
    # for i in range(10**8):
    #     i = i * 2
    ret = agent.query(data)
    if ret is None:
        return 'No workers available - cannot process the query.'
    return ret


if __name__ == '__main__':
    agent.start()
    app.run()
    agent.stop()
    agent.join()
