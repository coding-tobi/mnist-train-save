import * as d3 from "d3";

import ComponentDecorator from "../common/component-decorator";
import Component from "../common/component";

import CSS from "./training-chart-component.css";
import HTML from "./training-chart-component.html";

const LEFT = 45;
const RIGHT = 15;
const TOP = 15;
const BOTTOM = 30;

enum DataType {
  Batch = 1,
  Epoch,
}

interface BatchDataPoint {
  batch: number;
  acc: number;
}

interface EpochDataPoint extends BatchDataPoint {
  epoch: number;
}

interface Data<T extends BatchDataPoint = BatchDataPoint> {
  id: string;
  type: DataType;
  points: T[];
}

@ComponentDecorator({
  selector: "training-chart",
  style: CSS,
  template: HTML,
})
export default class TrainingChartComponent extends Component {
  private _batchData: Data<BatchDataPoint> = {
    id: "batch-acc",
    type: DataType.Batch,
    points: [{ batch: 0, acc: 0 }],
  };
  private _epochData: Data<EpochDataPoint> = {
    id: "epoch-val-acc",
    type: DataType.Epoch,
    points: [],
  };
  private render: (() => void) | null = null;

  public pushBatchData(data: BatchDataPoint, keep: number) {
    this._batchData.points.push(data);
    this._batchData.points = this._batchData.points.slice(
      Math.max(0, this._batchData.points.length - keep - 1),
      this._batchData.points.length
    );
    this.render?.call(this);
  }

  public pushEpochData(data: EpochDataPoint) {
    this._epochData.points.push(data);
    this.render?.call(this);
  }

  protected connected() {
    let curWidth = 0;
    let curHeight = 0;

    const resizeLoop = () => {
      const clientRect = this.getBoundingClientRect();
      if (clientRect.width !== curWidth || clientRect.height !== curHeight) {
        curWidth = clientRect.width;
        curHeight = clientRect.height;
        this.initD3(clientRect.width, clientRect.height);
      }

      if (this.isConnected) {
        setTimeout(resizeLoop, 150);
      }
    };
    resizeLoop();
  }

  private initD3(width: number, height: number) {
    if (width < LEFT + RIGHT) width = LEFT + RIGHT;
    if (height < TOP + BOTTOM) height = TOP + BOTTOM;

    const chartWidth = width - LEFT - RIGHT;
    const chartHeight = height - TOP - BOTTOM;

    // setup svg
    const svg = d3
      .select(this.getChildById("chart"))
      .attr("height", height + "px")
      .attr("width", width + "px")
      .attr("viewBox", `-${LEFT} -${TOP} ${width} ${height}`);

    svg.selectAll("g").remove();

    // create y-axis
    const yScale = d3
      .scalePow()
      .exponent(5)
      .range([chartHeight, 0])
      .domain([0, 1]);
    const yAxis = d3
      .axisLeft<number>(yScale)
      .tickFormat((d) => (d < 0.5 ? "" : Math.round(d * 100) + "%"));
    svg.append("g").call(yAxis);

    // tooltip stuff
    const tooltip = d3.select(this.getChildById("tooltip")).style("opacity", 0);
    const dotMouseOver = (e: MouseEvent, d: EpochDataPoint) => {
      tooltip.select("#epoch").text(d.epoch);
      tooltip.select("#acc").text((d.acc * 100.0).toString().slice(0, 5) + "%");
      tooltip
        .style("left", e.clientX + "px")
        .style("top", e.clientY + "px")
        .transition()
        .duration(150)
        .style("opacity", 1.0);
    };
    const dotMouseOut = () => {
      tooltip.transition().delay(1000).duration(300).style("opacity", 0);
    };

    this.render = () => {
      // calculate x-scales
      const batchXScale = d3
        .scaleLinear()
        .range([0, chartWidth])
        .domain(
          <[number, number]>d3.extent(this._batchData.points, (x) => x.batch)
        );
      const epochXScale = d3
        .scaleLinear()
        .range([0, chartWidth])
        .domain(
          <[number, number]>d3.extent(this._epochData.points, (x) => x.epoch)
        );

      // draw x-axis
      const xAxis = d3.axisBottom(batchXScale);
      svg.select(".x-axis").remove();
      svg
        .append("g")
        .classed("x-axis", true)
        .attr("transform", `translate(0, ${chartHeight})`)
        .call(xAxis);

      // draw lines
      const line = (type: DataType) =>
        d3
          .line<BatchDataPoint>()
          .x(
            (d) =>
              (type === DataType.Batch
                ? batchXScale(d.batch)
                : epochXScale((<EpochDataPoint>d).epoch)) || 0
          )
          .y((d) => yScale(d.acc) || 0);
      const lines = svg
        .selectAll<SVGPathElement, Data>(".line")
        .data([this._batchData, this._epochData], (d) => d.id);
      lines
        .enter()
        .append("path")
        .classed("line", true)
        .classed("batch-line", (d) => d.type === DataType.Batch)
        .classed("epoch-line", (d) => d.type === DataType.Epoch)
        .merge(lines)
        .attr("d", (d) => line(d.type)(d.points));
      lines.exit().remove();

      // draw epoch dots
      const dots = svg
        .selectAll<SVGCircleElement, EpochDataPoint>(".dot")
        .data(this._epochData.points);
      dots
        .enter()
        .append("circle")
        .classed("dot", true)
        .attr("r", "5")
        .on("mouseover", <any>dotMouseOver)
        .on("mouseout", <any>dotMouseOut)
        .merge(dots)
        .attr("cx", (d) => epochXScale(d.epoch) || 0)
        .attr("cy", (d) => yScale(d.acc) || 0);
      dots.exit().remove();
    };
  }
}
