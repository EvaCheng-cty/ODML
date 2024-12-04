import coremltools as ct
import pickle as pkl

model = pkl.load(open("svc_model_moredata.pkl", "rb"))

coreml_model = ct.converters.sklearn.convert(model)
coreml_model.save("svc_model_moredata.mlpackage")

fpath = "../rf models/"
for model_name in ["rf_100.pkl", "rf_500.pkl", "rf_1000.pkl"]:
    model = pkl.load(open(f"{fpath}/{model_name}", "rb"))
    coreml_model = ct.converters.sklearn.convert(model)
    coreml_model.save(f"{fpath}/{model_name.replace('.pkl', '.mlpackage')}")
