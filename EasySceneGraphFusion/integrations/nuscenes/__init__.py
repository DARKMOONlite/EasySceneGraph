"""NuScenes annotation, database, and temporal stabilisation workflows.

Modules:
    database: SQLite database access and row helpers.
    geometry: Camera projection, annotation conversion, and visualisation.
    import_graphs: Match dataset annotations to scene graphs and store matches.
    stabilise: Track and compress sample/sweep graphs using database windows.
"""
