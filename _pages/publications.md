---
layout: page
permalink: /publications/
title: Publications
description: Check the full publications <b><a href='https://scholar.google.com/citations?hl=ko&user=G3xl1HAAAAAJ&view_op=list_works&sortby=pubdate'>here</a></b>.
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
