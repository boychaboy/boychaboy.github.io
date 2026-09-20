---
layout: page
permalink: /publications/
title: Publications
description: Explore my complete publication record on <b><a href='https://scholar.google.com/citations?user=G3xl1HAAAAAJ&hl=en'>Google Scholar</a></b> · <b>558 citations</b> as of September 2026.
years: [2024, 2022, 2021]
nav: true
nav_order: 2
---
<!-- _pages/publications.md -->
<div class="publications">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>
