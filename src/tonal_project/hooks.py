# Importation des composants nécessaires de Kedro et PySpark
from kedro.framework.hooks import hook_impl
from pyspark import SparkConf
from pyspark.sql import SparkSession

class SparkHooks:
    @hook_impl
    def after_context_created(self, context) -> None:
        """
        Initialise une SparkSession en utilisant la configuration définie dans le dossier conf du projet.
        
        Paramètres:
        context: Le contexte Kedro, contenant la configuration du projet.
        """

        # Charger la configuration Spark à partir de spark.yaml en utilisant le chargeur de configuration du contexte
        parameters = context.config_loader["spark"]
        spark_conf = SparkConf().setAll(parameters.items())

        # Initialiser la session Spark avec la configuration chargée
        spark_session_conf = (
            SparkSession.builder.appName(context.project_path.name)
            .enableHiveSupport()
            .config(conf=spark_conf)
        )
        _spark_session = spark_session_conf.getOrCreate()

        # Définir le niveau de log de Spark à "WARN" pour réduire la verbosité des logs
        _spark_session.sparkContext.setLogLevel("WARN")
