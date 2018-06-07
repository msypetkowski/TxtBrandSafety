#!/usr/bin/env python3
from flask import request, Flask

from agents.front_agent import FrontAgent

app = Flask(__name__)
global_agent = FrontAgent()


@app.route('/compat', methods=['GET', 'POST'])
def calc_compatibility():
    data = list(request.form.keys())[0]
    # for i in range(10**8):
    #     i = i * 2
    ret = global_agent.query(data)
    if ret is None:
        return 'No workers available - cannot process the query.'
    return ret


if __name__ == '__main__':
    global_agent.start()
    app.run()
    global_agent.stop()
    global_agent.join()
