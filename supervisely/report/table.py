# example from here
# https://stackoverflow.com/questions/38893448/pagination-on-pandas-dataframe-to-html/49134917
base_html = """
<!doctype html>
<html><head>
<meta http-equiv="Content-type" content="text/html; charset=utf-8">
<script type="text/javascript" src="https://ajax.googleapis.com/ajax/libs/jquery/2.2.2/jquery.min.js"></script>
<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.16/css/jquery.dataTables.css">
<script type="text/javascript" src="https://cdn.datatables.net/1.10.16/js/jquery.dataTables.js"></script>
<link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons">

<style id="compiled-css" type="text/css">
      * {box-sizing: border-box;}

.wrapper {
    border: 2px solid #f76707;
    border-radius: 5px;
    background-color: #fff4e6;
}

.wrapper > div {
    border: 2px solid #ffa94d;
    border-radius: 5px;
    background-color: #ffd8a8;
    padding: 1em;
    color: #d9480f;
}
.wrapper {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  grid-gap: 10px;
}

  </style>

</head><body>%s
</body></html>
"""

#<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
#<script src="https://kit.fontawesome.com/a076d05399.js"></script>

table_html = """
<div>
    %s<script type="text/javascript">$(document).ready(function(){$('table').DataTable({"pageLength": 15 });});</script>
</div>
"""


div_wrapper = """
<div class="wrapper">{}</div>
"""

def df_html(df, index=True):
    return table_html % df.to_html(index=index, escape=False)


def compile_report(dfs, html_divs=[], index=True):
    result = ""
    for df in dfs:
        result += "{}\n".format(df_html(df, index))

    divs = ""
    for elem in html_divs:
        divs += "{}\n".format(elem)

    elmenets = div_wrapper.format(divs)
    result += elmenets
    return base_html % result

