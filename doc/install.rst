:orphan:

Installation
============

.. raw:: html

    <style>
        #right-column {
            max-width: 1000px;
        }
        .breadcrumb {
            display: none;
        }
        h1 {
            text-align: center;
            margin-bottom: 15px;
        }
        p.lead.grey-text {
            margin-bottom: 30px;
        }
        .footer-relations {
            border-top: 0px;
        }
        pre {
            background-color: #FFFFFF00;
            border-style: hidden;
            font-size: 87.5% !important;
            margin: 0 !important;
            padding: 0;
        }
        code {
            margin: 0 !important;
            padding: 0 40px 0 40px !important;
        }
    </style>

    <div class="container" id="main-column">
        <div class="text-center">
            <p class="lead grey-text w-responsive mx-auto">
                Strawberry Fields supports Python 3.7 or newer.
            </p>
            <p class="lead grey-text w-responsive mx-auto mb-6">
                If you currently do not have Python 3 installed, we recommend
                <a href="https://www.anaconda.com/download/">Anaconda for Python 3</a>,
                a distributed version of Python packaged for scientific computation.
            </p>
        </div>

        <ul class="picker nav nav-pills nav-justified mt-2" id="version">
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
    # Install the latest released version of Strawberry Fields.
    pip install strawberryfields --upgrade
                    </code>
                </pre>
            </div>
            <div class="tab-pane slide" id="preview" role="tabpanel">
                <pre>
                    <code class="bash">
    # Install the latest development version of Strawberry Fields.
    pip install git+https://github.com/XanaduAI/strawberryfields.git#egg=strawberryfields
                    </code>
                </pre>
            </div>
            <div class="tab-pane slide" id="source" role="tabpanel">
                <pre>
                    <code class="bash">
    # Download and install the latest source code from GitHub.
    git clone https://github.com/XanaduAI/strawberryfields.git
    cd strawberryfields
    pip install -e .
                    </code>
                </pre>
            </div>
        </div>
    </div>

    <script type="text/javascript">
        $(function(){
            let params = new URLSearchParams(window.location.search);
            let current_version = params.get("version");

            if (current_version) {
                $("#version li a").removeClass("active");
                $("#tab-version .tab-pane").removeClass("active");
                $("a[href='#" + current_version + "']").addClass("active");
                $("#" + current_version).show();
            };

            $("#version .nav-item a").click(function (e) {
                const new_version = this.hash.substr(1);
                if (current_version != new_version) {
                    $("#" + current_version).hide();
                    $("#" + new_version).show();

                    params.set("version", new_version);
                    const new_rel_path_query = window.location.pathname + "?" + params.toString();
                    history.pushState(null, "", new_rel_path_query);
                };
            });

            // Change active navbar element to "Install".
            $(".nav-item.active").removeClass("active");
            $(".nav-item a:contains('Install')").parent().addClass("active");
        });
    </script>
