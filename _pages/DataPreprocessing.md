---
layout: archive
permalink: /data-preprocessing/
title: "Projects"
author_profile: true
header:
  overlay_image	: "/images/main.jpg"
  excerpt : Currently working on Disaster Detection using Image Analysis
---

{% include group-by-array collection=site.posts field="tags" %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}
