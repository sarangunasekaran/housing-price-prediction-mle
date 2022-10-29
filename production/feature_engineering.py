"""Processors for the feature engineering step of the worklow.

The step loads cleaned training data, processes the data for outliers,
missing values and any other cleaning steps based on business rules/intuition.

The trained pipeline and any artifacts are then saved to be used in
training/scoring pipelines.
"""
import logging
import os.path as op

from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ta_lib.core.api import (
    get_dataframe,
    get_feature_names_from_column_transformer,
    get_package_path,
    load_dataset,
    register_processor,
    save_pipeline,
    DEFAULT_ARTIFACTS_PATH
)

from ta_lib.data_processing.api import Outlier

logger = logging.getLogger(__name__)


@register_processor("feat-engg", "transform-features")
def transform_features(context, params):
    """Transform dataset to create training datasets."""

    input_features_ds = "train/sales/features"
    input_target_ds = "train/sales/target"

    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    # load datasets
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)

    cat_columns = train_X.select_dtypes("object").columns
    num_columns = train_X.select_dtypes("number").columns

    # Treating Outliers
    outlier_transformer = Outlier(method=params["outliers"]["method"])
    train_X = outlier_transformer.fit_transform(
        train_X, drop=params["outliers"]["drop"]
    )

    # NOTE: You can use ``Pipeline`` to compose a collection of transformers
    # into a single transformer. In this case, we are composing a
    # ``TargetEncoder`` and a ``SimpleImputer`` to first encode the
    # categorical variable into a numerical values and then impute any missing
    # values using ``most_frequent`` strategy.
    tgt_enc_simple_impt = Pipeline(
        [
            ("target_encoding", TargetEncoder(return_df=False)),
            ("simple_impute", SimpleImputer(strategy="most_frequent")),
        ]
    )

    # NOTE: the list of transformations here are not sequential but weighted
    # (if multiple transforms are specified for a particular column)
    # for sequential transforms use a pipeline as shown above.
    features_transformer = ColumnTransformer(
        [
            # categorical columns
            (
                "tgt_enc",
                TargetEncoder(return_df=False),
                list(
                    set(cat_columns)
                    - set(["technology", "functional_status", "platforms"])
                ),
            ),
            (
                "tgt_enc_sim_impt",
                tgt_enc_simple_impt,
                ["technology", "functional_status", "platforms"],
            ),
            # numeric columns
            ("med_enc", SimpleImputer(strategy="median"), num_columns),
        ]
    )

    # Check if the data should be sampled. This could be useful to quickly run
    # the pipeline for testing/debugging purposes (undersample)
    # or profiling purposes (oversample).
    # The below is an example how the sampling can be done on the train data if required.
    # Model Training in this reference code has been done on complete train data itself.
    sample_frac = params.get("sampling_fraction", None)
    if sample_frac is not None:
        logger.warn(f"The data has been sample by fraction: {sample_frac}")
        sample_X = train_X.sample(frac=sample_frac, random_state=context.random_seed)
    else:
        sample_X = train_X
    sample_y = train_y.loc[sample_X.index]


    # Train the feature engg. pipeline prepared earlier. Note that the pipeline is
    # fitted on only the **training data** and not the full dataset.
    # This avoids leaking information about the test dataset when training the model.
    # In the below code train_X, train_y in the fit_transform can be replaced with
    # sample_X and sample_y if required. 
    train_X = get_dataframe(
        features_transformer.fit_transform(train_X, train_y),
        get_feature_names_from_column_transformer(features_transformer),
    )

    # Note: we can create a transformer/feature selector that simply drops
    # a specified set of columns. But, we don't do that here to illustrate
    # what to do when transformations don't cleanly fall into the sklearn
    # pattern.
    curated_columns = list(
        set(train_X.columns.to_list())
        - set(
            [
                "manufacturer",
                "inventory_id",
                "ext_grade",
                "source_channel",
                "tgt_enc_iter_impt_platforms",
                "ext_model_family",
                "order_no",
                "line",
                "inventory_id",
                "gp",
                "selling_price",
                "selling_cost",
                "invoice_no",
                "customername",
            ]
        )
    )

    # saving the list of relevant columns and the pipeline.
    save_pipeline(
        curated_columns, op.abspath(op.join(artifacts_folder, "curated_columns.joblib"))
    )
    save_pipeline(
        features_transformer, op.abspath(op.join(artifacts_folder, "features.joblib"))
    )