## config.json format


**name** (`required`) - String

>  Item name



**description** - String

>  Item description



**type** (`required`) - String

>  Item type
>
>  *available values:* 
>  ```
>  project | model | app | notebook | plugin | project
>  ```



**categories** - Array[String]
>  List of item categories
>
>  *example:*
>
>  ```
>  ["Demo"]
>  ```



**icon** - String
>  Link to Item icon



**icon_background** - String
>  Icon background color in css format
>
>  *example:*
>  ```
>  "rgb(32, 29, 102)"
>  ```



**icon_cover** - Boolean
>  Fit icon to available size



<!-- "docker_image": "docker.deepsystems.io/supervisely/five/python-runner:ecosystem_add_images_project", -->

**main_script** (`required`) - String
>  > Only for items with type "app"
>
>  *example:*
>  ```
>  "src/add_images_project.py",
>  ```



**gui_template** - String
>  > Only for items with type "app"
>
>  Path to application UI template
>
>  *example:*
>  ```
>  "src/gui.html",
>  ```



**modal_template** - String
>  > Only for items with type "app"
>
>  Path to template that will be shown in "Run Application" dialog
>
>  *example:*
>  ```
>  "src/modal.html",
>  ```



**modal_template_state** - Object
>  > Only for items with type "app"
>
>  > Required if modal_template is specified
>
>  Properties that will be availble in "Run Application" dialog
>
>  *example:*
>  ``` json
>  {
>    "teamId": null,
>    "workspaceId": null,
>    "projectName": ""
>  }
>  ```



**task_location** (`required`) - String
>  > Only for items with type "app"
>
>  Specify where application session will be displayed
>
>  *available values:*
>    
>      workspace_tasks | application_sessions



**context_menu** - Object
>  > Only for items with type "app"
>
>  Display application in context menu of specified entities
> 
>   - **target** (`required`) - Array(String)
>
>      Entities list where application will be shown
>
>     *available values:*
>   
>         images_project | videos_project | point_cloud_project | images_dataset | videos_dataset | point_cloud_dataset
>
>   
> - **context_root** - String
>
>     Root element in context menu
>
>     *available values:*
>     
>       Download as | Run App (`default`) | Report
>
>
> - **context_category** - String
>
>     Subcategory in context menu
>
> *example*: 
>  ``` json
>  {
>    "target": ["images_project", "images_dataset"],
>    "context_root": "Report",
>    "context_category": "Objects"
>  }
>  ```



**headless** - Boolean
>  > Only for items with type "app"
