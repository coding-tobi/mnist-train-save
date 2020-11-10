import * as tf from "@tensorflow/tfjs";
import * as d3 from "d3";
import * as moment from "moment";

import ComponentDecorator from "../common/component-decorator";
import Component from "../common/component";
import MnistDataLoader from "../mnist-data-loader";
import TrainingChartComponent from "../training-chart-component/training-chart-component";

import CSS from "./mnist-trainer-component.css";
import HTML from "./mnist-trainer-component.html";

const TRAIN_IMAGES_URL = "ressources/train-images-idx3-ubyte.gz";
const TRAIN_LABELS_URL = "ressources/train-labels-idx1-ubyte.gz";
const TEST_IMAGES_URL = "ressources/t10k-images-idx3-ubyte.gz";
const TEST_LABELS_URL = "ressources/t10k-labels-idx1-ubyte.gz";

const BATCH_SIZE = 512;
const VALIDATION_BATCH_SIZE = 4096;
const EPOCHS = 5;

@ComponentDecorator({
  selector: "mnist-trainer",
  style: CSS,
  template: HTML,
})
export default class MnistTrainerComponent extends Component {
  private _trainDataset: MnistDataLoader;
  private _testDataset: MnistDataLoader;
  private _model: tf.LayersModel;

  private _batchesPerEpoch = 0;
  private _curBatch = 0;
  private _batchCounter = 0;
  private _curEpoch = 0;
  private _curLoss = 0;
  private _curAcc = 0;
  private _curValLoss = 0;
  private _curValAcc = 0;
  private _trainingStarted = false;
  private _trainStartTime = moment();

  constructor() {
    super();
    this._trainDataset = new MnistDataLoader(
      TRAIN_IMAGES_URL,
      TRAIN_LABELS_URL
    );
    this._testDataset = new MnistDataLoader(TEST_IMAGES_URL, TEST_LABELS_URL);
    this._model = this.createModel();
  }

  protected connected() {
    this.render();
    const startButton = this.getChildById("start-button");
    startButton.onclick = async () => {
      startButton.remove();
      this._trainStartTime = moment();
      this._trainingStarted = true;
      const trainingChart = this.getChildById<TrainingChartComponent>("chart");
      await this._trainDataset.load();
      await this._testDataset.load();

      this._batchesPerEpoch = Math.floor(
        this._trainDataset.numberOfImages / BATCH_SIZE
      );

      const trainDataset = tf.data.func(() =>
        this._trainDataset.nextBatch(BATCH_SIZE)
      );
      const valData = this._testDataset.nextBatch(VALIDATION_BATCH_SIZE).value;
      let batchOffset = 0;

      this._model.fitDataset(trainDataset, {
        epochs: EPOCHS,
        validationData: [valData.xs, valData.ys],
        callbacks: {
          onBatchEnd: (batch, logs) => {
            this._curBatch = batch + 1;
            this._curLoss = logs?.["loss"] || 0;
            this._curAcc = logs?.["acc"] || 0;

            this._batchCounter = this._curBatch + batchOffset;
            trainingChart.pushBatchData(
              {
                batch: this._batchCounter,
                acc: logs?.["acc"] || 0,
              },
              this._batchesPerEpoch
            );
            this.render();
          },
          onEpochBegin: () => {
            batchOffset = this._batchCounter;
          },
          onEpochEnd: (epoch, logs) => {
            console.log("EPOCH END!", epoch, logs);
            console.log("tensors", tf.memory().numTensors);
            this._curEpoch = epoch + 1;
            this._curValLoss = logs?.["val_loss"] || 0;
            this._curValAcc = logs?.["val_acc"] || 0;

            trainingChart.pushEpochData({
              batch: this._batchCounter,
              acc: logs?.["val_acc"] || 0,
              epoch: this._curEpoch,
            });
            this.render();
          },
          onTrainEnd: () => {
            const saveButton = document.createElement("button");
            saveButton.textContent = "save model";
            saveButton.className = "flat-btn";
            saveButton.onclick = () => {
              this._model.save("downloads://mnist-cnn-model");
            };
            this.getChildById("progress-card-footer").appendChild(saveButton);
          },
        },
      });
    };
  }

  protected disconnected() {}

  private render() {
    const curSample = this._batchCounter * BATCH_SIZE;
    const samples = this._trainDataset.numberOfImages * EPOCHS;

    const numberFormat = d3.format(".3f");
    const siNumberFormat = d3.format(".3s");
    const percentageFormat = d3.format(".2%");
    this.select("batch-header")
      .datum({
        curBatch: this._curBatch,
        batchesPerEpoch: this._batchesPerEpoch,
      })
      .text((d) => `Batch (${d.curBatch} / ${d.batchesPerEpoch})`);
    this.select("epoch-header")
      .datum({
        curEpoch: this._curEpoch,
        epochs: EPOCHS,
      })
      .text((d) => `Epoch (${d.curEpoch} / ${d.epochs})`);
    this.select("loss-text")
      .datum(this._curLoss)
      .text((d) => numberFormat(d));
    this.select("acc-text")
      .datum(this._curAcc)
      .text((d) => percentageFormat(d));
    this.select("val-loss-text")
      .datum(this._curValLoss)
      .text((d) => numberFormat(d));
    this.select("val-acc-text")
      .datum(this._curValAcc)
      .text((d) => percentageFormat(d));

    this.select("samples-text")
      .datum({
        curSample,
        samples,
      })
      .text(
        (d) => `${siNumberFormat(d.curSample)} / ${siNumberFormat(d.samples)}`
      );
    this.select("time-text")
      .datum({
        started: this._trainingStarted,
        fromNow: this._trainStartTime.fromNow(),
      })
      .text((d) => (d.started ? d.fromNow : "not yet"));
    this.select("remaining-text")
      .datum({
        started: 0 < curSample,
        remaining: moment.duration(
          (moment().diff(this._trainStartTime) / curSample) *
            (samples - curSample)
        ),
      })
      .text((d) => (d.started ? d.remaining.humanize() : "no idea"));
  }

  private select(id: string) {
    return d3.select(this.getChildById(id));
  }
  private createModel() {
    const model = tf.sequential();
    model.add(
      tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 5,
        filters: 48,
        strides: 1,
        activation: "relu",
        kernelInitializer: "varianceScaling",
      })
    );
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
    model.add(
      tf.layers.conv2d({
        kernelSize: 5,
        filters: 96,
        strides: 1,
        activation: "relu",
        kernelInitializer: "varianceScaling",
      })
    );
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
    model.add(tf.layers.flatten());

    model.add(tf.layers.dropout({ rate: 0.5 }));
    model.add(
      tf.layers.dense({
        units: 128,
        kernelInitializer: "varianceScaling",
        activation: "relu",
      })
    );
    model.add(tf.layers.dropout({ rate: 0.5 }));
    model.add(
      tf.layers.dense({
        units: 10,
        kernelInitializer: "varianceScaling",
        activation: "softmax",
      })
    );

    const optimizer = tf.train.adam();
    model.compile({
      optimizer: optimizer,
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });

    return model;
  }
}
