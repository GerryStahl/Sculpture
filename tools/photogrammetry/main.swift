// photogrammetry — Apple Object Capture CLI (the engine behind PhotoCatch)
// Usage: photogrammetry <frames_folder> <output.usdz> [detail] [ordering]
//   detail:   preview | reduced | medium | full | raw   (default: full)
//   ordering: sequential | unordered                    (default: sequential)
// Frames from a turntable video are consecutive views, so "sequential" is the right default.

import Foundation
import RealityKit

func die(_ msg: String) -> Never {
    fputs("error: \(msg)\n", stderr)
    exit(1)
}

let args = CommandLine.arguments
guard args.count >= 3 else {
    die("Usage: photogrammetry <frames_folder> <output.usdz> [detail] [ordering]")
}

let inputURL = URL(fileURLWithPath: args[1], isDirectory: true)
let outputURL = URL(fileURLWithPath: args[2])
let detailName = args.count > 3 ? args[3] : "full"
let orderingName = args.count > 4 ? args[4] : "sequential"

let detail: PhotogrammetrySession.Request.Detail
switch detailName {
case "preview": detail = .preview
case "reduced": detail = .reduced
case "medium": detail = .medium
case "full": detail = .full
case "raw": detail = .raw
default: die("Unknown detail level: \(detailName)")
}

var configuration = PhotogrammetrySession.Configuration()
configuration.sampleOrdering = orderingName == "unordered" ? .unordered : .sequential
configuration.featureSensitivity = .normal
configuration.isObjectMaskingEnabled = true

let session: PhotogrammetrySession
do {
    session = try PhotogrammetrySession(input: inputURL, configuration: configuration)
} catch {
    die("Cannot start session: \(error)")
}

do {
    try session.process(requests: [.modelFile(url: outputURL, detail: detail)])
} catch {
    die("Cannot process request: \(error)")
}

do {
    for try await output in session.outputs {
        switch output {
        case .requestProgress(_, let fraction):
            print(String(format: "progress %.0f%%", fraction * 100))
        case .requestComplete(_, .modelFile(let url)):
            print("model \(url.path)")
        case .requestError(_, let error):
            die("Reconstruction failed: \(error)")
        case .invalidSample(let id, let reason):
            fputs("warning: invalid sample \(id): \(reason)\n", stderr)
        case .skippedSample(let id):
            fputs("warning: skipped sample \(id)\n", stderr)
        case .processingComplete:
            exit(0)
        case .processingCancelled:
            die("Processing cancelled")
        default:
            break
        }
    }
} catch {
    die("Session failed: \(error)")
}
