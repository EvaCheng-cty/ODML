import coremltools as ct
import pickle as pkl

model = pkl.load(open("svc_model_moredata.pkl", "rb"))

coreml_model = ct.converters.sklearn.convert(model)
coreml_model.save("svc_model_moredata.mlpackage")