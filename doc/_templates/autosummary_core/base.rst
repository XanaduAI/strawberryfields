{% if referencefile %}
.. include:: {{ referencefile }}
{% endif %}

{% if module.split(".")[1:] | length >= 1 %}
	{% set mod = module.split(".")[1:] | join(".") %}
	{% set mod = "sf." + mod %}
{% else %}
	{% set mod = "sf" %}
{% endif %}

{{ mod }}.{{ objname }}
={% for i in range(mod|length) %}={% endfor %}{{ underline }}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}
