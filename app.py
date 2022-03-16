import json
import os
import subprocess

import numpy as np
import optuna
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import streamlit as st
from genericpath import exists
from tensorflow.keras.losses import mean_squared_error as mse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["NUMEXPR_MAX_THREADS"] = "16"

# MySQL setting
import configparser

config = configparser.ConfigParser()
config.read("./config.ini")

mysqlconf = "MySQL"
user_name = config.get(mysqlconf, "user")
pass_word = config.get(mysqlconf, "password")
host_name = config.get(mysqlconf, "host")


st.set_page_config(layout="wide")


@st.cache
def read_file(file):
    df = pd.read_csv(file)
    return df


study_dir = "study"  # result will be saved under "study" folder
try:
    os.mkdir(study_dir)
except FileExistsError:
    pass


st.title("Neural Network to simulate 2-D measurements")

st.sidebar.write("step0a: Project name (working folder)")
wdirin = st.sidebar.text_input(
    label="if not specified, filename is used for foldername", value=None
)
st.sidebar.write(" ")
st.sidebar.write("step0b: upload teacher data csv:")
inputf = st.sidebar.file_uploader(label="csv!", type=["csv"])
if inputf is None:
    inputf = st.sidebar.text_input(
        label="Or specify name of teacher data csv.\nWrite 'train_df.csv' to use existing work",
        value="",
    )
    inputf = wdirin + "/" + inputf
    if inputf == wdirin + "/":
        inputf = None
st.write("input: ", inputf)
st.header("STEP1: data_preparation")

if inputf is not None:
    if wdirin == "None":
        wdir = inputf.name.split(".csv")[0]
    else:
        wdir = wdirin

    wdir = os.path.join("study", wdir)

    try:
        os.mkdir(wdir)
    except FileExistsError:  # remove old files
        pass

    path_setting = os.path.join(wdir, "NNsetting.json")
    if exists(path_setting):
        with open(path_setting) as f:
            NNsetting = json.load(f)
        X1_name = NNsetting["X1_name"]
        X2_name = NNsetting["X2_name"]
        Y_name = NNsetting["Y_name"]
        n_trial_ini = NNsetting["n_trial"]
        n_cores_ini = NNsetting["n_cores"]
    else:
        X1_name = None
        X2_name = None
        Y_name = None
        n_trial_ini = 10
        n_cores_ini = 5


with st.expander("click to expand data_preparation"):
    if inputf is not None:
        df = read_file(inputf)
        st.dataframe(df)
        collist = list(df.columns)
        if X1_name is not None:
            j = 0
            for i in collist:
                if i == X1_name:
                    break
                j += 1
            X1_name = st.selectbox("Choose X1 axis(feature 1) :", collist, index=j)
        else:
            X1_name = st.selectbox("Choose X1 axis(feature 1) :", collist)
        if X2_name is not None:
            j = 0
            for i in collist:
                if i == X2_name:
                    break
                j += 1
            X2_name = st.selectbox("Choose X2 axis(feature 2) :", collist, index=j)
        else:
            X2_name = st.selectbox("Choose X2 axis(feature 2) :", collist)
        if Y_name is not None:
            j = 0
            for i in collist:
                if i == Y_name:
                    break
                j += 1
            Y_name = st.selectbox("Choose Y axis(target) :", collist, index=j)
        else:
            Y_name = st.selectbox("Choose Y axis(target) :", collist)
        st.write("X1 is : ", X1_name)
        st.write("X2 is : ", X2_name)
        st.write("Y is : ", Y_name)

        N_COLORS = len(df[X2_name].unique())
        colormap = sns.color_palette("gist_rainbow", N_COLORS).as_hex()

        if st.checkbox("rawdata_plot"):
            fig = px.line(
                data_frame=df,
                x=X1_name,
                y=Y_name,
                color=X2_name,
                color_discrete_sequence=colormap,
            )
            st.plotly_chart(fig)

        X2list = df[X2_name].unique()
        if st.checkbox("X2_selecion"):
            st.write("unique X2-axis values are: ")
            st.write(str(X2list))

            selected_2nd_axis = st.multiselect(
                "select 2nd-axis-values for teacher", X2list, default=X2list
            )

            st.write(str(selected_2nd_axis))

            train_df = df[df[X2_name].isin(selected_2nd_axis)]
            fig = px.line(
                data_frame=train_df,
                x=X1_name,
                y=Y_name,
                color=X2_name,
                color_discrete_sequence=colormap,
            )
            st.plotly_chart(fig)
        else:
            selected_2nd_axis = X2list
            train_df = df[df[X2_name].isin(selected_2nd_axis)]

        path_train = os.path.join(wdir, "train_df.csv")
        train_df.to_csv(path_train, index=False)

        y_train = train_df[Y_name]
        mean_y_orig = np.mean(y_train)
        sdev_y_orig = np.std(y_train, ddof=0)

        if st.checkbox("image plot of traindata"):
            path_train = os.path.join(wdir, "train_df.csv")
            df_train = pd.read_csv(path_train)
            traindata = go.Heatmap(
                z=df_train[Y_name].T, x=df_train[X1_name], y=df_train[X2_name]
            )
            layouttrain = go.Layout(
                title="image plot of train data",
                xaxis=dict(ticks=""),
                yaxis=dict(ticks="", nticks=0, scaleanchor="x"),
            )
            figtrain = go.Figure(data=traindata, layout=layouttrain)
            st.plotly_chart(figtrain)


# --- model construction part ---
st.header("STEP2: model_construction")
with st.expander("click to expand model_construction"):
    st.header("Learning conditions")
    if inputf is not None:

        n_trial = st.number_input(
            label="hyperparam optimization total trial number",
            value=n_trial_ini,
            min_value=1,
        )
        n_cores = st.number_input(
            label="# of cores used parallel, make sure not to exceed total PC cores",
            value=n_cores_ini,
            min_value=1,
        )
        solve_option = st.selectbox(
            "choose solver",
            (
                "Adaptive Momentum Optimization",
                "Root Mean Square Propagation",
                "Nesterov's Accelarated Gradient",
                "Stochastic Gradient Descent",
            ),
        )

        if solve_option == "Stochastic Gradient Descent":
            NNsetting = dict(
                X1_name=X1_name,
                X2_name=X2_name,
                Y_name=Y_name,
                n_trial=n_trial,
                n_cores=n_cores,
                solver="SGD",
                # lr = lr,
                nesterov=False,
                momentum=0.0,
                mean_y_orig=mean_y_orig,
                sdev_y_orig=sdev_y_orig,
            )
        elif solve_option == "Nesterov's Accelarated Gradient":
            NNsetting = dict(
                X1_name=X1_name,
                X2_name=X2_name,
                Y_name=Y_name,
                n_trial=n_trial,
                n_cores=n_cores,
                solver="SGD",
                # lr = lr,
                nesterov=True,
                momentum=0.01,
                mean_y_orig=mean_y_orig,
                sdev_y_orig=sdev_y_orig,
            )
        elif solve_option == "Root Mean Square Propagation":
            NNsetting = dict(
                X1_name=X1_name,
                X2_name=X2_name,
                Y_name=Y_name,
                n_trial=n_trial,
                n_cores=n_cores,
                solver="RMSprop",
                # lr = lr,
                # rho = 0.9,
                # epsilon = None,
                # decay = 0.0
                mean_y_orig=mean_y_orig,
                sdev_y_orig=sdev_y_orig,
            )

        elif solve_option == "Adaptive Momentum Optimization":
            NNsetting = dict(
                X1_name=X1_name,
                X2_name=X2_name,
                Y_name=Y_name,
                n_trial=n_trial,
                n_cores=n_cores,
                solver="Adam",
                # lr = lr,
                # epsilon = None,
                # decay = 0.0
                mean_y_orig=mean_y_orig,
                sdev_y_orig=sdev_y_orig,
            )

        path_setting = os.path.join(wdir, "NNsetting.json")
        save_setting = st.button("save_settings")
        if save_setting:
            with open(path_setting, "w") as f:
                json.dump(NNsetting, f, ensure_ascii=False)
            st.write(f"{path_setting} saved")

        update_setting = st.sidebar.button("update settings")
        if update_setting:
            with open(path_setting, "w") as f:
                json.dump(NNsetting, f, ensure_ascii=False, indent=4)
        if exists(path_setting):
            with open(path_setting) as f:
                NNsetting = json.load(f)
            st.sidebar.write(NNsetting)

        st.write("")
        st.write("--caution-- this may take time, check settings")
        learn_exe = st.button("execute NNlearning")
        # check_b = st.button("check if learning finished")
        if learn_exe:
            check_load_study = False

            cmd = ["python", "learn_parent.py", wdir, "True"]
            proc = subprocess.Popen(cmd)
            if proc.poll() is None:
                st.write("go to STEP2.5, check learning process")
            else:
                st.write("study done!")


# --- model construction part ---
st.header("STEP2.5: check_learning_process")
with st.expander("click to expand check_learning_process"):

    if "rangel" not in st.session_state:
        st.session_state["rangel"] = 0.995
    else:
        range_l = st.number_input(
            label="lower range of graph for r2_score graph below",
            key="range_l",
            value=st.session_state.rangel,
            format="%.4f",
        )
        st.session_state["rangel"] = range_l

    check_hist = st.button("check/update record")

    if check_hist:
        study_name = "optuna_NNtemp"
        storage = (
            "mysql+pymysql://"
            + user_name
            + ":"
            + pass_word
            + "@"
            + host_name
            + "/NN_interp"
        )
        study = optuna.load_study(study_name=study_name, storage=storage)
        dfcheckrec = study.trials_dataframe()
        path_dfcheckrec = os.path.join(wdir, "df_study_trials.csv")
        dfcheckrec.to_csv(path_dfcheckrec, index=False)
        checkrec_n = (dfcheckrec["state"] == "COMPLETE").sum()  # number of "complete"
        if checkrec_n == 0:
            st.write("wait until any trial COMPLETEs, to see learning curves")
        dfcheckrec = pd.read_csv(path_dfcheckrec)
        dfcheckrec = dfcheckrec.reindex(
            columns=[
                "state",
                "value",
                "datetime_start",
                "datetime_complete",
                "duration",
            ]
        )
        check_fin = (dfcheckrec["state"] == "RUNNING").sum()  # number of "running"
        fin_trial_n = (dfcheckrec["state"] == "COMPLETE").sum() + (
            dfcheckrec["state"] == "PRUNED"
        ).sum()
        if check_fin < 1:
            st.write("learning finished!")
        else:
            st.write("learning on going,", fin_trial_n, "trials finished so far")
        if "dfcheckrec_show" not in st.session_state:  # initialize
            st.session_state["dfcheckrec_show"] = 1

        if st.session_state["dfcheckrec_show"] is not None:
            path_dfcheckrec = os.path.join(wdir, "df_study_trials.csv")
            dfcheckrec = pd.read_csv(path_dfcheckrec)
            dfcheckrec = dfcheckrec.reindex(
                columns=[
                    "state",
                    "value",
                    "datetime_start",
                    "datetime_complete",
                    "duration",
                ]
            )
            st.dataframe(dfcheckrec)

        checkrec_n = (dfcheckrec["state"] == "COMPLETE").sum()  # number of "complete"
        if checkrec_n > 0:
            study_name = "optuna_NNtemp"
            storage = (
                "mysql+pymysql://"
                + user_name
                + ":"
                + pass_word
                + "@"
                + host_name
                + "/NN_interp"
            )
            study = optuna.load_study(study_name=study_name, storage=storage)
            dfcheckrec = study.trials_dataframe()

            last_trial = int(str(study.trials[-1]).split("number=")[1].split(",")[0])
            trial_num = last_trial
            dftemp2 = pd.DataFrame(
                columns=["epochs", "loss", "trial_n", "lr_n", "batch_n"]
            )
            for i in range(trial_num + 1):
                val = study.trials[i].intermediate_values
                xval = val.keys()
                xlist = list(xval)
                yval = val.values()
                ylist = list(yval)
                trial_n = [i] * len(ylist)
                lr_n = [
                    float(str(study.trials[i]).split("'lr_ini': ")[1].split(",")[0])
                ] * len(ylist)
                batch_temp = str(study.trials[i]).split("'batch': ")[1].split(",")[0]
                if "}" in batch_temp:
                    batch_temp = batch_temp.split("}")[0]
                batch_n = [int(batch_temp)] * len(ylist)
                runstate = [str(study.trials[i].state).split(".")[1]] * len(ylist)

                dftemp = pd.DataFrame(
                    list(zip(xlist, ylist, trial_n, lr_n, batch_n, runstate)),
                    columns=[
                        "epochs",
                        "r2_score",
                        "trial_n",
                        "lr_n",
                        "batch_n",
                        "runstate",
                    ],
                )

                dftemp2 = pd.concat([dftemp2, dftemp], axis=0)

            fig_study = px.line(
                dftemp2,
                x="epochs",
                y="r2_score",
                color="trial_n",
                hover_data=["lr_n", "batch_n", "runstate"],
                log_y=False,
            )
            y_min = max(dftemp2["r2_score"])  # max for R2_score

            lny_max = 1.0001

            lny_min = range_l

            best_trial_n = int(str(study.best_trial).split("number=")[1].split(",")[0])
            best_loss = y_min
            best_trial_lr = study.best_params["lr_ini"]
            best_trial_batch = study.best_params["batch"]
            st.write(
                "best r2_score",
                y_min,
                "on trial:",
                best_trial_n,
                ", lr :",
                best_trial_lr,
                ", batch:",
                best_trial_batch,
            )

            fig_study.update_layout(
                yaxis_range=[lny_min, lny_max], width=650, height=500
            )
            st.plotly_chart(fig_study)

    st.write("you can choose:")
    continue_study = st.button("keep search with optuna")
    if continue_study:
        check_load_study = False
        cmd = ["python", "learn_parent.py", wdir, "False"]
        proc = subprocess.Popen(cmd)
        if proc.poll() is None:
            st.write("check learning process again")
        else:
            st.write("study done!")

    st.write("or")
    check_result = st.checkbox("check_study_result")
    if check_result:

        import joblib

        path_study = os.path.join(wdir, "study.pkl")
        study = joblib.load(path_study)
        st.write(
            "minimum loss of ", study.best_value, "achieved on the following conditions"
        )
        st.write(study.best_params)

        if st.checkbox("visualize learned curves"):
            path_train = os.path.join(wdir, "train_df.csv")
            path_fitted = os.path.join(wdir, "df_fitted.csv")
            train_df = pd.read_csv(path_train)
            df_fitted = pd.read_csv(path_fitted)
            N_COLORS = len(train_df[NNsetting["X2_name"]].unique())
            colormap = sns.color_palette("gist_rainbow", N_COLORS).as_hex()
            fig = px.line(
                data_frame=train_df,
                x=NNsetting["X1_name"],
                y=NNsetting["Y_name"],
                color=NNsetting["X2_name"],
                color_discrete_sequence=colormap,
            )
            fig.add_scatter(
                x=df_fitted[NNsetting["X1_name"]],
                y=df_fitted[NNsetting["Y_name"]],
                mode="markers",
                marker=dict(color="black", symbol="circle-open"),
                name="learned",
            )
            st.plotly_chart(fig)

            ytrain = train_df[NNsetting["Y_name"]]
            yfit = df_fitted[NNsetting["Y_name"]]

        if st.checkbox("image plot"):
            path_fitted = os.path.join(wdir, "df_fitted.csv")
            df_fitted = pd.read_csv(path_fitted)
            fitdata = go.Heatmap(
                z=df_fitted[NNsetting["Y_name"]].T,
                x=df_fitted[NNsetting["X1_name"]],
                y=df_fitted[NNsetting["X2_name"]],
            )
            layoutim = go.Layout(
                title="image plot of fitted data",
                xaxis=dict(ticks=""),
                yaxis=dict(ticks="", nticks=0, scaleanchor="x"),
            )
            figim = go.Figure(data=fitdata, layout=layoutim)
            st.plotly_chart(figim)


st.header("STEP3: use_model")
with st.expander("click to expand use_model"):
    if inputf is not None:
        path_setting = os.path.join(wdir, "NNsetting.json")
        path_train = os.path.join(wdir, "train_df.csv")

        # get default values for prediction X1,X2 range&steps
        if os.path.exists(path_train) and os.path.exists(path_setting):
            with open(path_setting) as f:
                NNsetting = json.load(f)
            df = pd.read_csv(path_train)
            X1_min = min(df[NNsetting["X1_name"]])
            X1_max = max(df[NNsetting["X1_name"]])
            X2_min = min(df[NNsetting["X2_name"]])
            X2_max = max(df[NNsetting["X2_name"]])
            X1_step = float(df[NNsetting["X1_name"]][1] - df[NNsetting["X1_name"]][0])
            if X1_step < 0.1:
                X1_step = 0.2
            X2_step = abs(
                float(
                    df[NNsetting["X2_name"]].unique()[1]
                    - df[NNsetting["X2_name"]].unique()[0]
                )
            )
            if X2_step < 0.1:
                X2_step = 0.1

            st.header("Prediction")
            st.write("learned range of X1", X1_min, "to", X1_max)
            n1 = st.number_input(
                label="predict range: X1 minimum", key="n1", value=round(X1_min, 1)
            )
            X1pred_min = n1

            n2 = st.number_input(
                label="predict range: X1 maximum", key="n2", value=round(X1_max, 1)
            )
            X1pred_max = n2

            n3 = st.number_input(
                label="predict range: X1 step", key="n3", value=round(X1_step, 1)
            )
            X1pred_step = n3

            st.write("learned range of X2", X2_min, "to", X2_max)
            n4 = st.number_input(
                label="predict range: X2 minimum", key="n4", value=round(X2_min, 1)
            )
            X2pred_min = n4

            n5 = st.number_input(
                label="predict range: X2 maximum", key="n5", value=round(X2_max, 1)
            )
            X2pred_max = n5

            n6 = st.number_input(
                label="predict range: X2 step", key="n6", value=round(X2_step, 2)
            )
            X2pred_step = n6

            st.write("")
            st.write(
                "X1 predict: from", X1pred_min, "to", X1pred_max, "step:", X1pred_step
            )
            selected_1st_axis = np.arange(
                X1pred_min, X1pred_max + X1pred_step, X1pred_step
            )
            st.write("X1 will be:", str(selected_1st_axis))
            st.write("")
            st.write(
                "X2 predict: from", X2pred_min, "to", X2pred_max, "step:", X2pred_step
            )
            selected_2nd_axis = np.arange(
                X2pred_min, X2pred_max + X2pred_step, X2pred_step
            )
            st.write("X2 will be:", str(selected_2nd_axis))

            dfs = []
            if len(selected_1st_axis) > 0 and len(selected_2nd_axis) > 0:
                for element in selected_2nd_axis:
                    df = pd.DataFrame(
                        data=selected_1st_axis, columns=[NNsetting["X1_name"]]
                    )
                    df[[NNsetting["X2_name"]]] = element
                    dfs.append(df)
                final = pd.concat(dfs)
                final.reset_index(drop=True, inplace=True)
                path_pred_feat = os.path.join(wdir, "pred_feature.csv")
                final.to_csv(path_pred_feat, index=False)

            simulate_b = st.button("simulate")
            if simulate_b:
                import simulate

                simulate.main(wdir)

            path_sim = os.path.join(wdir, "df_simulated.csv")
            if os.path.exists(path_sim):
                show_simulation = st.checkbox("show simulated result")
                if show_simulation:
                    df_simulated = pd.read_csv(path_sim)
                    N_COLORS = len(df_simulated[NNsetting["X2_name"]].unique())
                    colormap = sns.color_palette("gist_rainbow", N_COLORS).as_hex()
                    fig = px.line(
                        data_frame=df_simulated,
                        x=NNsetting["X1_name"],
                        y=NNsetting["Y_name"],
                        color=NNsetting["X2_name"],
                        color_discrete_sequence=colormap,
                    )
                    st.plotly_chart(fig)

                if st.checkbox("image plot of simulated result"):
                    df_simulated = pd.read_csv(path_sim)
                    simdata = go.Heatmap(
                        z=df_simulated[NNsetting["Y_name"]].T,
                        x=df_simulated[NNsetting["X1_name"]],
                        y=df_simulated[NNsetting["X2_name"]],
                    )
                    layoutsim = go.Layout(
                        title="image plot of simulated data",
                        xaxis=dict(ticks=""),
                        yaxis=dict(ticks="", nticks=0, scaleanchor="x"),
                    )
                    figsim = go.Figure(data=simdata, layout=layoutsim)
                    st.plotly_chart(figsim)
            if os.path.exists(path_sim):
                with open(path_sim) as f:
                    st.download_button(
                        "Download df_simulated.csv", f, file_name="df_simulated.csv"
                    )
