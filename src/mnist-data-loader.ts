import * as tf from "@tensorflow/tfjs";
import { Tensor4D } from "@tensorflow/tfjs";
import { Inflate } from "pako";

const IMAGE_DATA_START = 16;
const LABEL_DATA_START = 8;

export interface MnistBatch {
  done: boolean;
  value: {
    xs: tf.Tensor4D;
    ys: tf.Tensor2D;
  };
}

export default class MnistDataLoader {
  private static readonly _NORMALIZE_FACTOR = tf.scalar(1.0 / 255.0, "float32");

  private _imageData: Uint8Array | null = null;
  private _labelData: Uint8Array | null = null;
  private _numberOfImages = 0;
  private _numberOfRows = 0;
  private _numberOfColumns = 0;

  private _imageDataSize = 0;
  private _cursor = 0;
  private _indices: number[] = [];

  public get numberOfImages() {
    return this._numberOfImages;
  }

  public get numberOfRows() {
    return this._numberOfRows;
  }

  public get numberOfColumns() {
    return this._numberOfColumns;
  }

  public async load(imagesUrl: string, labelsUrl: string): Promise<void> {
    const [imagesResponse, labelsResponse] = await Promise.all([
      fetch(imagesUrl),
      fetch(labelsUrl),
    ]);

    if (imagesResponse.body && labelsResponse.body) {
      [this._imageData, this._labelData] = await Promise.all([
        MnistDataLoader.inflate(imagesResponse.body),
        MnistDataLoader.inflate(labelsResponse.body),
      ]);
      this._numberOfImages = MnistDataLoader.readUInt32(4, this._imageData);
      this._numberOfRows = MnistDataLoader.readUInt32(8, this._imageData);
      this._numberOfColumns = MnistDataLoader.readUInt32(12, this._imageData);
      this._imageDataSize = this._numberOfRows * this._numberOfColumns;
      this._indices = [...new Array(this._numberOfImages).keys()];
    } else {
      return Promise.reject("LOADING DATA FAILED!");
    }
  }

  public nextBatch(size: number): MnistBatch {
    console.assert(this._labelData, "NO DATA LOADED!");
    console.assert(this._imageData, "NO DATA LOADED!");

    return tf.tidy(() => {
      const labelData = <Uint8Array>this._labelData;
      const imageData = <Uint8Array>this._imageData;
      const batch = [...Array(size).keys()]
        .map((i) => (i + this._cursor) % this._numberOfImages)
        .map((i) => this._indices[i])
        .map((i) => [
          i + LABEL_DATA_START,
          i * this._imageDataSize + IMAGE_DATA_START,
        ])
        .map(([labelIndex, imageIndex]) => ({
          label: labelData[labelIndex],
          imageData: imageData.slice(
            imageIndex,
            imageIndex + this._imageDataSize
          ),
        }));

      const xsData = new Array<number>().concat.apply(
        [],
        batch.map((x) => Array.from(x.imageData))
      );
      const xs = tf
        .tensor4d(
          xsData,
          [size, this._numberOfRows, this._numberOfColumns, 1],
          "float32"
        )
        .mul<Tensor4D>(MnistDataLoader._NORMALIZE_FACTOR);

      const ysData = batch.map((y) => y.label);
      const ys = tf.oneHot(ysData, 10).as2D(ysData.length, 10);

      const done = this._numberOfImages <= this._cursor + size;
      this._cursor = (this._cursor + size) % this._numberOfImages;

      if (done) this.shuffle();

      return {
        done,
        value: {
          xs,
          ys,
        },
      };
    });
  }

  /**
   * Performs a Fisher–Yates shuffle on this._indices!
   */
  private shuffle() {
    for (let i = 0; i < this._indices.length - 1; i++) {
      const other = i + Math.floor(Math.random() * (this._indices.length - i));
      const temp = this._indices[i];
      this._indices[i] = this._indices[other];
      this._indices[other] = temp;
    }
  }

  /**
   * inflate a stream with the pako Inflator!
   * @param stream compressed Data
   */
  private static async inflate(
    stream: ReadableStream<Uint8Array>
  ): Promise<Uint8Array> {
    const reader = stream.getReader();
    const inflator = new Inflate();

    for (;;) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      } else if (value) {
        inflator.push(value);
      }
    }

    return inflator.result as Uint8Array;
  }

  private static readUInt32(position: number, data: Uint8Array) {
    let result = 0;
    for (let i = 0; i < 4; i++) {
      result = (result << 8) + data[i + position];
    }
    return result;
  }
}
