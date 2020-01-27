---
layout: archive
permalink: /data-preprocessing/
title: "Projects"
excerpt: *"The only way to do great work is to love what you do"*
author_profile: true
header:
  overlay_image	: "/images/moun.jpg"



---

{% include group-by-array collection=site.posts field="tags" %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}
