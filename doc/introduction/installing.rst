Installation
============

.. raw:: html

    <style>
    .nav-link {
        color: black;
    }

    .picker {
        box-shadow: 0 2px 5px 0 rgba(0,0,0,.16),0 2px 10px 0 rgba(0,0,0,.12);
        border-radius: 10px 10px 0 0;
        border-collapse: collapse;
        overflow: hidden;
    }

    .nav-pills .active {
        border-radius: 0px;
        background-color: #19b392 !important;
    }

    .tab-content {
        margin-top: -23px;
    }

    .nav-item input {
        opacity: 0;
        margin-left: -10px;
    }

    label {
        margin-bottom: unset;
    }

    .nav-link.active:hover {
        color: white !important;
    }

    .tab-pane pre code {
        padding: 0px 40px 0px 40px !important;
        font-size: 120% !important;
        box-shadow: 0 2px 5px 0 rgba(0,0,0,.16),0 2px 10px 0 rgba(0,0,0,.12);
        border-radius: 0 0 10px 10px;
    }

    pre {
        margin: 0;
        border: none;
        background-color: unset;
        padding: 0;
    }

    code {
        margin: 0;
        padding: 0;
    }
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.15.10/styles/tomorrow-night.min.css">

    <div class="container <!--mt-5 pt-5-->" id="main-column">
        <div class="text-center">
            <!--<h1>Install StrawberryFields</h1>-->
            <p class="lead grey-text w-responsive mx-auto">PennyLane supports Python 3.5 or newer.</p>
            <p class="lead grey-text w-responsive mx-auto mb-5">If you currently do not have Python 3 installed, we recommend <a href="https://www.anaconda.com/download/">Anaconda for Python 3</a>, a distributed version of Python packaged for scientific computation.</p>
        </div>

        <ul class="picker nav nav-pills nav-justified mt-5" id="version">
            <li class="nav-item">
                <a class="nav-link active" data-toggle="tab" href="#stable" role="tab">Stable</a>
            </li>
            <li class="nav-item">
                <a class="nav-link" data-toggle="tab" href="#preview" role="tab">Preview</a>
            </li>
            <li class="nav-item">
                <a class="nav-link" data-toggle="tab" href="#source" role="tab">Source</a>
            </li>
        </ul>

        <!-- Tab panels -->
        <div class="tab-content pt-0" id="tab-version">

            <div class="tab-pane in show active" id="stable" role="tabpanel">
                <pre>
                    <code class="bash">
    # install the latest released version of PennyLane
    pip install pennylane --upgrade
                    </code>
                </pre>
            </div>
            <div class="tab-pane slide" id="preview" role="tabpanel">
                <pre>
                    <code class="bash">
    # install the latest development version of PennyLane
    pip install git+https://github.com/XanaduAI/pennylane.git#egg=pennylane
                    </code>
                </pre>
            </div>
            <div class="tab-pane slide" id="source" role="tabpanel">
                <pre>
                    <code class="bash">
    # download and install the latest source code from GitHub
    git clone https://github.com/XanaduAI/pennylane.git
    cd pennylane
    pip install -e .
                    </code>
                </pre>
            </div>
        </div>
    </div>

    <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.15.10/highlight.min.js"></script>
    <script>hljs.initHighlightingOnLoad();</script>