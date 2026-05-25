import ROOT
from registry import register_routine

@register_routine("drawTTgrid")
def drawTTgrid(hist, c):
    c.cd()
    lines = []
    line_color = ROOT.kBlack
    line_style = 1
    line_width = 1
    x_min = hist.GetXaxis().GetXmin()
    x_max = hist.GetXaxis().GetXmax()
    y_min = hist.GetYaxis().GetXmin()
    y_max = hist.GetYaxis().GetXmax()

    # vertical grid lines
    for i in range(1, 66, 5):
      x = i - 0.5
      line = ROOT.TLine(x, y_min, x, y_max)
      line.SetLineColor(line_color)
      line.SetLineStyle(line_style)
      line.SetLineWidth(line_width)
      line.Draw("same")
      lines.append(line)

    # horizontal grid lines
    for j in range(1, 11, 5):
      y = j - 0.5
      line = ROOT.TLine(x_min, y, x_max, y)
      line.SetLineColor(line_color)
      line.SetLineStyle(line_style)
      line.SetLineWidth(line_width)
      line.Draw("same")
      lines.append(line)
    return lines
