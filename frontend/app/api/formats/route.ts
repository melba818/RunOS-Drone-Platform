import { NextResponse } from 'next/server'

// Supported annotation formats for computer vision datasets
const formats = [
  {
    id: 'yolo',
    name: 'YOLO',
    extensions: ['.txt'],
    task: ['detection', 'segmentation']
  },
  {
    id: 'coco',
    name: 'COCO JSON',
    extensions: ['.json'],
    task: ['detection', 'segmentation', 'keypoints']
  },
  {
    id: 'pascal-voc',
    name: 'Pascal VOC',
    extensions: ['.xml'],
    task: ['detection', 'segmentation']
  },
  {
    id: 'labelme',
    name: 'LabelMe',
    extensions: ['.json'],
    task: ['detection', 'segmentation', 'polygon']
  },
  {
    id: 'tensorflow-csv',
    name: 'TensorFlow CSV',
    extensions: ['.csv'],
    task: ['detection']
  },
  {
    id: 'createml',
    name: 'CreateML',
    extensions: ['.json'],
    task: ['detection', 'classification']
  }
]

export async function GET() {
  return NextResponse.json({ formats })
}
