import os,json,uproot,argparse,sys,ROOT
import numpy as np
import array
import glob
from math import sqrt


def has_branch(fname, branch):
    f = ROOT.TFile.Open(fname)
    if not f or f.IsZombie():
        return False
    t = f.Get("tree")
    if not t:
        return False

    return t.GetBranchStatus(branch)


def gaussFit(h,name,Run,output_dir, xmin=-1, xmax=-1):

    x = ROOT.RooRealVar(f"x_{name}_{Run}","E/E_{True}",
                        h.GetXaxis().GetXmin(),
                        h.GetXaxis().GetXmax())

    data = ROOT.RooDataHist(f"data_{name}_{Run}", "data",ROOT.RooArgList(x),h)

    peak = h.GetBinCenter(h.GetMaximumBin())

    mean  = ROOT.RooRealVar(f"mean_{name}", "Gaussian mean",peak,peak-3,peak+3)

    sigma = ROOT.RooRealVar(f"sigma_{name}", "Gaussian sigma",h.GetRMS(),0.1*h.GetRMS(),5*h.GetRMS())

    gauss = ROOT.RooGaussian(f"gauss_{name}", "Gaussian",x,mean,sigma)

    nsig = ROOT.RooRealVar(f"nsig_{name}", "signal yield",h.Integral(),0.0,10.0*h.Integral())

    model = ROOT.RooAddPdf(f"model_{name}_{Run}", "extended Gaussian model",ROOT.RooArgList(gauss),ROOT.RooArgList(nsig))

    fitArgs = [
        ROOT.RooFit.Extended(True),
        ROOT.RooFit.Save(),
        ROOT.RooFit.PrintLevel(-1)
    ]

    if xmin >= 0 and xmax >= 0:
        fitArgs.insert(0, ROOT.RooFit.Range("fitRange"))
        x.setRange("fitRange", xmin, xmax)

    result = model.fitTo(data, *fitArgs)

    canvas = ROOT.TCanvas()

    frame = x.frame()
    data.plotOn(frame)

    if xmin >= 0 and xmax >= 0:
        model.plotOn(frame,
                   ROOT.RooFit.Range("fitRange"),
                   ROOT.RooFit.NormRange("fitRange"))
    else:
        model.plotOn(frame)

    frame.Draw()

    chi2 = frame.chiSquare()

    pt = ROOT.TPaveText(0.60, 0.65, 0.88, 0.88, "NDC")
    pt.SetFillColor(0)
    pt.SetTextFont(42)
    pt.SetBorderSize(0)
    pt.SetTextSize(0.05)

    pt.AddText(f"m_{{core}} = {mean.getVal():.3g} #pm {mean.getError():.3g}")
    pt.AddText(f"#sigma_{{core}} = {sigma.getVal():.3g} #pm {sigma.getError():.3g}")
    pt.AddText(f"#chi^2_{{core}} = {chi2:.3g}" )

    pt.Draw()

    canvas.Update()

    filename = f"fit_gauss_run_{Run}"
    output_path = os.path.join(output_dir, filename)
    canvas.SaveAs(output_path + ".pdf")
    canvas.SaveAs(output_path + ".root")

    return {
        "mean": (mean.getVal(), mean.getError()),
        "sigma": (sigma.getVal(), sigma.getError())
    }

def cbFit(h,name,Run,output_dir,xmin=-1,xmax=-1):

    x = ROOT.RooRealVar(f"x_{name}_{Run}", "E/E_{True}", h.GetXaxis().GetXmin(), h.GetXaxis().GetXmax())

    data = ROOT.RooDataHist(f"data_{name}_{Run}", "data", ROOT.RooArgList(x), h)

    peak = h.GetBinCenter(h.GetMaximumBin())

    mean  = ROOT.RooRealVar(f"mean_{name}", "DCB mean",peak,peak-3,peak+3)

    sigma = ROOT.RooRealVar(f"sigma_{name}", "DCB sigma",h.GetRMS(),0.1*h.GetRMS(),5*h.GetRMS())

    alphaL = ROOT.RooRealVar(f"alphaL_{name}", "alphaL", 1.5, 0.1, 5.0)
    nL     = ROOT.RooRealVar(f"nL_{name}",     "nL",     3.0, 0.5, 20.0)

    alphaR = ROOT.RooRealVar(f"alphaR_{name}", "alphaR", 1.5, 0.1, 5.0)
    nR     = ROOT.RooRealVar(f"nR_{name}",     "nR",     3.0, 0.5, 20.0)

    dcb = ROOT.RooCrystalBall(f"dcb_{name}", "Double Crystal Ball",x,mean,sigma,alphaL, nL,alphaR, nR)

    nsig = ROOT.RooRealVar(f"nsig_{name}", "signal yield",h.Integral(),0.0,10.0*h.Integral())
    model = ROOT.RooAddPdf(f"model_{name}_{Run}", "extended DCB model",ROOT.RooArgList(dcb),ROOT.RooArgList(nsig))

    fitArgs = [
        ROOT.RooFit.Extended(True),
        ROOT.RooFit.Save(),
        ROOT.RooFit.PrintLevel(-1)
    ]

    if xmin >= 0 and xmax >= 0:
        fitArgs.insert(0, ROOT.RooFit.Range("fitRange"))
        x.setRange("fitRange", xmin, xmax)

    result = model.fitTo(data, *fitArgs)

    canvas = ROOT.TCanvas()

    frame = x.frame()
    data.plotOn(frame)
    model.plotOn(frame, ROOT.RooFit.Range("fitRange"),ROOT.RooFit.NormRange("fitRange"))

    frame.Draw()

    chi2 = frame.chiSquare()

    pt = ROOT.TPaveText(0.60, 0.65, 0.88, 0.88, "NDC")
    pt.SetFillColor(0)
    pt.SetTextFont(42)
    pt.SetBorderSize(0)
    pt.SetTextSize(0.05)

    pt.AddText(f"m_{{core}} = {mean.getVal():.3g} #pm {mean.getError():.3g}")
    pt.AddText(f"#sigma_{{core}} = {sigma.getVal():.3g} #pm {sigma.getError():.3g}")
    pt.AddText(f"#chi^2_{{core}} = {chi2:.3g}" )

    pt.Draw()

    canvas.Update()

    filename = f"fit_dcb_run_{Run}"
    output_path = os.path.join(output_dir, filename)
    canvas.SaveAs(output_path + ".pdf")
    canvas.SaveAs(output_path + ".root")

    return {
        "mean": (mean.getVal(), mean.getError()),
        "sigma": (sigma.getVal(), sigma.getError())
    }


def lognFit(h, name, Run, output_dir, xmin=-1, xmax=-1):

    x = ROOT.RooRealVar(f"x_{name}_{Run}", "E/E_{True}",
                        h.GetXaxis().GetXmin(),
                        h.GetXaxis().GetXmax())

    data = ROOT.RooDataHist(f"data_{name}_{Run}", "data",
                            ROOT.RooArgList(x), h)

    peak  = h.GetBinCenter(h.GetMaximumBin())
    sigma0 = h.GetRMS()

    eta   = ROOT.RooRealVar(f"eta_{name}",   "eta",   0.1, 0.01, 1.0)
    sigma = ROOT.RooRealVar(f"sigma_{name}", "sigma", sigma0, 0.1, 10000)
    mean  = ROOT.RooRealVar(f"mean_{name}",  "peak",  peak, peak-3*sigma0, peak+3*sigma0)
    amp = ROOT.RooRealVar("amp", "amplitude", 0.3*h.Integral(), 0, 1e7)

    sqrt2pi = ROOT.RooConstVar("sqrt2pi", "sqrt2pi", (2*3.14159265)**0.5)
    c235    = ROOT.RooConstVar("c235", "2.35 const", 2.35)

    expr = "x[4] * (x[1] / (x[5] * x[2] * ((2/x[6]) * log(x[1]*x[6]/2 + sqrt(1 + pow(x[1]*x[6]/2,2)))))) * exp(-0.5 * pow(log(max(1e-4, 1 - (x[1]/x[2])*(x[0] - x[3])))/((2/x[6]) * log(x[1]*x[6]/2 + sqrt(1 + pow(x[1]*x[6]/2,2)))) ,2))"

    logn_pdf = ROOT.RooGenericPdf(f"logn_{name}","log-normal-like",expr,ROOT.RooArgList(x, eta, sigma, mean, amp, sqrt2pi,c235))
    nsig = ROOT.RooRealVar(f"nsig_{name}", "signal yield",h.Integral(),0.0,10.0*h.Integral())
    model = ROOT.RooAddPdf(f"model_{name}_{Run}", "extended logn model",ROOT.RooArgList(logn_pdf),ROOT.RooArgList(nsig))

    fitArgs = [
        ROOT.RooFit.Extended(True),
        ROOT.RooFit.Save(),
        ROOT.RooFit.PrintLevel(-1)
    ]

    if xmin >= 0 and xmax >= 0:
        x.setRange("fitRange", xmin, xmax)
        fitArgs.insert(0, ROOT.RooFit.Range("fitRange"))

    result = model.fitTo(data, *fitArgs)

    canvas = ROOT.TCanvas()
    frame = x.frame()

    data.plotOn(frame)
    model.plotOn(frame,ROOT.RooFit.Range("fitRange"),ROOT.RooFit.NormRange("fitRange"))

    frame.Draw()

    chi2 = frame.chiSquare()

    pt = ROOT.TPaveText(0.60, 0.65, 0.88, 0.88, "NDC")
    pt.SetFillColor(0)
    pt.SetTextFont(42)
    pt.SetBorderSize(0)
    pt.SetTextSize(0.05)

    pt.AddText(f"Peak = {mean.getVal():.3g} #pm {mean.getError():.3g}")
    pt.AddText(f"Sigma = {sigma.getVal():.3g} #pm {sigma.getError():.3g}")
    pt.AddText(f"#chi^2_{{core}} = {chi2:.3g}" )

    pt.Draw()

    canvas.Update()

    filename = f"fit_logn_run_{Run}"
    output_path = os.path.join(output_dir, filename)
    canvas.SaveAs(output_path + ".pdf")
    canvas.SaveAs(output_path + ".root")

    return {
        "mean": (mean.getVal(), mean.getError()),
        "sigma": (sigma.getVal(), sigma.getError())
    }


def main(arguments):

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-g",  f"--fit-type", type=str, required=True, help="fit type")
    parser.add_argument("-i",  f"--input-dir", type=str, required=True, help="input directory containing ROOT file with unpacked tree")
    parser.add_argument("-ro", f"--plot-output-dir", type=str, required=True, help="directory for output plots")
    parser.add_argument("-f", f"--fit-output-dir", type=str, required=True, help="directory for fits")
    parser.add_argument("-j", f"--run-info-json", type=str, required=False, help="run and energy sample")

    args = parser.parse_args(arguments)

    json_dict = json.load(open(args.run_info_json, "r"))
    fit_type=args.fit_type

    input_dir=args.input_dir
    plot_output_dir=args.plot_output_dir
    fit_output_dir=args.fit_output_dir
    os.makedirs(plot_output_dir, exist_ok=True)
    os.makedirs(fit_output_dir, exist_ok=True)

    Run=json_dict["global"]["run info"]["run list"]
    Ebins=json_dict["global"]["run info"]["run energies"]
    En,eEn,mu,emu,sigma,esigma = [],[],[],[],[],[]
    roofit_objects = []

    lin = ROOT.TGraphErrors(len(Ebins))
    fitlin = ROOT.TGraphErrors(len(Ebins))
    res = ROOT.TGraphErrors(len(Ebins))
    res2 = ROOT.TGraphErrors(len(Ebins))

    for ie in range(len(Ebins)):

        c = ROOT.TCanvas()
        c.SetGrid()

        run=Run[ie]
        energy=Ebins[ie]

        good_files = []

        charge_list = []
        peak_list = []

        chain = ROOT.TChain("tree")

        pattern = os.path.join(input_dir, f"run_{run}/{run}_*_reco.root") 

        for f in glob.glob(pattern): 
            if has_branch(f, "ecal_charge_sum_5x5"): 
                chain.Add(f)
            else:
                print("Skipping:", f) 

        print(f"Run {run}: added {chain.GetNtrees()} files")


        h = ROOT.TH1F(f"Charge_5x5_{run}", "", 1000, 0, 50000)
        h2=ROOT.TH2F(f"Peak_vs_Charge_seed_{run}","",500,0,50000,500,0,50000)

        chain.Draw(f"ecal_charge_sum_5x5>>Charge_5x5_{run}", "", "goff")

        chain.Draw(f"ecal_peak[ecal_seed_ch]:ecal_charge_seed>>Peak_vs_Charge_seed_{run}", "", "goff")

        h.Draw()

        h2.Draw()

        MinMaxX=np.zeros(2).astype(float)

        Rgx=np.array([0.08,0.99],float)
        h2.ProjectionX().GetQuantiles(2,MinMaxX,Rgx)

        xmin,xmax = MinMaxX
        Rgy=np.array([0.08,0.99],float)

        MinMaxY=np.zeros(2).astype(float)
        h2.ProjectionY().GetQuantiles(2,MinMaxY,Rgy)

        ymin,ymax = MinMaxY

        print("ranges (x, y)", xmin,xmax, ymin,ymax)
        h2.GetXaxis().SetRangeUser(xmin, xmax)
        h2.GetYaxis().SetRangeUser(ymin, ymax)
        h2.SetStats(0)
        h2.SetTitle("Peak vs Charge;Charge[ADC];Peak value [ADC]")
        h2.GetXaxis().SetRangeUser(xmin, xmax)
        h2.GetYaxis().SetRangeUser(ymin, ymax)
        h2.SetMarkerStyle(24)
        h2.SetMarkerSize(0.8)
        h2.SetMarkerColor(ROOT.kBlack)
        ROOT.gStyle.SetOptTitle(1)
        ROOT.gStyle.SetTitleAlign(23)
        ROOT.gStyle.SetTitleX(0.5)
        h2.Draw("COLZ")

        hprof=h2.ProfileX()
        hprof.Draw("same")

        fit = ROOT.TF1("fit", "pol1",xmin,xmax)
        hprof.Fit(fit,"R")

        slope = fit.GetParameter(1)
        chi2  = fit.GetChisquare()
        ndf = fit.GetNDF()

        pave = ROOT.TPaveText(0.15, 0.7, 0.35, 0.88, "NDC")
        pave.SetFillColor(0)
        pave.SetTextFont(42)
        pave.SetTextSize(0.03)
        pave.SetBorderSize(1)

        pave.AddText(f"Slope = {slope:.3f}")
        pave.AddText(f"#chi^2 = {chi2:.2f}")
        pave.AddText(f"Ndof = {ndf}")
        pave.Draw()
        fit.Draw("same")

        filename_h2 = f"PeakvsChargeFit_{run}"
        output_path_h2 = os.path.join(plot_output_dir, filename_h2)
        c.SaveAs(output_path_h2 + ".pdf")
        c.SaveAs(output_path_h2 + ".root")
        c.Clear()


        h.GetXaxis().SetTitle("Charge [ADC]")
        h.GetYaxis().SetTitle("Nevents")

#        print("Entries in chain:", chain.GetEntries())
#        print("DF snapshot entries:", df.Count().GetValue())

        print(run, h.Integral())

        max_bin = h.GetMaximumBin()
        max_position = h.GetBinCenter(max_bin)
        max_value = h.GetBinContent(max_bin)
        bin1 = h.FindFirstBinAbove(max_value/2)
        bin2 = h.FindLastBinAbove(max_value/2)
        fwhm = h.GetBinCenter(bin2) - h.GetBinCenter(bin1)

        min = max_position - 3*fwhm
        max = max_position + 2*fwhm

        if fit_type=="gauss":
            results = gaussFit(h,h.GetName(),run,fit_output_dir,min,max)
        if fit_type=="dcb":
            results = cbFit(h,h.GetName(),run,fit_output_dir,min,max)
        if fit_type=="logn":
            results = lognFit(h,h.GetName(),run,fit_output_dir,min,max)

        roofit_objects.append(results)

        mu_val, emu_val = results["mean"]
        sig_val, esig_val = results["sigma"]

        resolution_error = sqrt((esig_val/mu_val)**2+emu_val**2*(sig_val/mu_val**2)**2+5e-4**2)
        quotient_error = sqrt(((emu_val*(energy+55.5))/(mu_val)**2)**2+(3.25/mu_val)**2+(energy*0.05/mu_val)**2)

        print("Energy/Mean/eMean/Sigma/eSigma")
        print(energy,mu_val,emu_val,sig_val,esig_val)

        lin.SetPoint(ie, mu_val, energy)                 #plot energy linearity
        lin.SetPointError(ie,emu_val,energy*0.025)

        fitlin.SetPoint(ie, energy, (energy+55.5)/mu_val)       #plot y/x from last one (accounting for the fact that the energy linearity
        fitlin.SetPointError(ie,energy*0.025,quotient_error)               #fit has a different than zero constant term)

        res.SetPoint(ie,energy,sig_val)                  #plot sigma vs beam energy
        res.SetPointError(ie,energy*0.025,esig_val)

        res2.SetPoint(ie,energy,100*(sig_val/mu_val))    #plot resolution vs beam energy
        res2.SetPointError(ie,0,100*resolution_error)

    canvas = ROOT.TCanvas()
    canvas.SetGrid()

    lin.SetMarkerStyle(24)
    lin.SetMarkerSize(0.8)
    lin.SetMarkerColor(ROOT.kBlack)
    fitlin.SetMarkerStyle(24)
    fitlin.SetMarkerSize(0.8)
    fitlin.SetMarkerColor(ROOT.kBlack)
    res.SetMarkerStyle(24)
    res.SetMarkerSize(0.8)
    res.SetMarkerColor(ROOT.kBlack)
    res2.SetMarkerStyle(24)
    res2.SetMarkerSize(0.8)
    res2.SetMarkerColor(ROOT.kBlack)
    ROOT.gStyle.SetOptTitle(1)
    ROOT.gStyle.SetTitleAlign(23)
    ROOT.gStyle.SetTitleX(0.5)

####
    lin.SetTitle(f"Energy linearity ({fit_type} fit);Mu_charge_5x5 [ADC];Beam energy [GeV]")
    lin.Draw("AP")

    ROOT.gStyle.SetOptFit(111)
    fit = ROOT.TF1("fit", "pol1",0,25000)
    lin.Fit(fit,"R")
    canvas.Update()

    filename_lin = f"Energy_linearity_{fit_type}"
    output_path_lin = os.path.join(plot_output_dir, filename_lin)
    canvas.SaveAs(output_path_lin + ".pdf")
    canvas.SaveAs(output_path_lin + ".root")
    canvas.Clear()
####
    fitlin.SetTitle(f"Linear fit energy/Mu_charge_5x5 ({fit_type} fit);Beam energy [GeV]; energy/Mu_charge_5x5 [arbitrary units]")
    fitlin.Draw("AP")

    ROOT.gStyle.SetOptFit(111)
    fit = ROOT.TF1("fit", "pol1",0,130)
    fitlin.Fit(fit,"R")
    canvas.Update()

    filename_fitlin = f"Energy_linearity_linear_fit_{fit_type}"
    output_path_fitlin = os.path.join(plot_output_dir, filename_fitlin)
    canvas.SaveAs(output_path_fitlin + ".pdf")
    canvas.SaveAs(output_path_fitlin + ".root")
    canvas.Clear()
####
    res.SetTitle(f"Sigma vs Beam energy ({fit_type} fit);Beam energy [GeV];Sigma_charge_5x5 [ADC]")
    res.Draw("AP")
    filename_res = f"SigmavsBeamEn_{fit_type}"
    output_path_res = os.path.join(plot_output_dir, filename_res)
    canvas.SaveAs(output_path_res + ".pdf")
    canvas.SaveAs(output_path_res + ".root")
    canvas.Clear()
####
    res2.SetTitle(f"Resolution ({fit_type} fit);Beam energy[GeV];Sigma/Mu_(charge_5x5) [%]")
    res2.Draw("AP")

    ROOT.gStyle.SetOptFit(111)
    fit = ROOT.TF1("fit","sqrt( ([0]/sqrt(x))**2 + ([1]/x)**2 + [2]**2 )",0,130)
    fit.SetParLimits(0, 0, 100)
    fit.SetParLimits(1, 0, 100)
    fit.SetParLimits(2, 0, 10)
    res2.Fit(fit,"R")
    canvas.Update()

    filename_res2 = f"Resolution_{fit_type}"
    output_path_res2 = os.path.join(plot_output_dir, filename_res2)
    canvas.SaveAs(output_path_res2 + ".pdf")
    canvas.SaveAs(output_path_res2 + ".root")
####


    input("finito")
if __name__ == "__main__":
    main(sys.argv[1:])
