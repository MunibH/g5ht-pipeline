import QtQuick 2.6
import QtQuick.Window 2.2
import QtQuick.Controls 2.3
import QtQuick.Layouts 1.0
import org.julialang 1.0

ApplicationWindow {
	visible: true
	width: 1400
	height: 800
	title: qsTr("NIRTrack")
        onClosing: {
            Julia.window_close()
    }

	RowLayout {
		anchors.fill: parent
		JuliaCanvas {
			id: img_canvas
			paintFunction: cf_update
			Layout.alignment: Qt.AlignTop
			Layout.fillWidth: false
			Layout.fillHeight: false
			Layout.preferredWidth: 968
            Layout.preferredHeight: 732
		}

        
        RowLayout {
            spacing: 6
            Layout.alignment: Qt.AlignTop
            
            ColumnLayout {                
                Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "Speed (mm/s)\navg 1s"
                    font.pointSize: 10
                }
                Text {
                    id: text_speed_avg
	                Layout.alignment: Qt.AlignCenter
	                text: "N/A"
                    font.pointSize: 20
                }
                Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "Recording"
                    font.pointSize: 10
                }
                Text {
                    id: text_recording_duration
	                Layout.alignment: Qt.AlignCenter
	                text: "N/A"
                    font.pointSize: 20
                }
                Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "stage x"
                    font.pointSize: 10
                }
                Text {
                    id: stage_x
	                Layout.alignment: Qt.AlignCenter
	                text: "N/A"
                    font.pointSize: 15
                }
                Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "stage y"
                    font.pointSize: 10
                }
                Text {
                    id: stage_y
	                Layout.alignment: Qt.AlignCenter
	                text: "N/A"
                    font.pointSize: 15
                }
				Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "TL x"
                    font.pointSize: 10
                }
                Text {
                    id: tl_x
	                Layout.alignment: Qt.AlignCenter
	                text: "N/A"
                    font.pointSize: 15
                }
				Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "TL y"
                    font.pointSize: 10
                }
                Text {
                    id: tl_y
	                Layout.alignment: Qt.AlignCenter
	                text: "N/A" 
                    font.pointSize: 15
                }
                
                Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "TR x"
                    font.pointSize: 10
                }
                Text {
                    id: tr_x
	                Layout.alignment: Qt.AlignCenter
	                text: "N/A"
                    font.pointSize: 15
                }
				Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "TR y"
                    font.pointSize: 10
                }
                Text {
                    id: tr_y
	                Layout.alignment: Qt.AlignCenter
	                text: "N/A" 
                    font.pointSize: 15
                }
                
                Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "BR x"
                    font.pointSize: 10
                }
                Text {
                    id: br_x
	                Layout.alignment: Qt.AlignCenter
	                text: "N/A"
                    font.pointSize: 15
                }
				Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "BR y"
                    font.pointSize: 10
                }
                Text {
                    id: br_y
	                Layout.alignment: Qt.AlignCenter
	                text: "N/A" 
                    font.pointSize: 15
                }
                
                Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "BL x"
                    font.pointSize: 10
                }
                Text {
                    id: bl_x
	                Layout.alignment: Qt.AlignCenter
	                text: "N/A"
                    font.pointSize: 15
                }
				Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "BL y"
                    font.pointSize: 10
                }
                Text {
                    id: bl_y
	                Layout.alignment: Qt.AlignCenter
	                text: "N/A" 
                    font.pointSize: 15
                }
                Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "Distance to patch (mm)"
                    font.pointSize: 10
                }
                Text {
                    id: d2p
	                Layout.alignment: Qt.AlignCenter
	                text: "N/A" 
                    font.pointSize: 20
                }
                
            }
        }
        
        RowLayout {
            spacing: 6
            Layout.alignment: Qt.AlignTop
            
            ColumnLayout {
				Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "Tracking"
                    font.pointSize: 10
                }
                Button {
                id: button_tracking
                Layout.alignment: Qt.AlignCenter
                text: "Start"
                font.pointSize: 10
                onClicked: { button_tracking.text = Julia.toggle_tracking().toString() }
            	}
            	Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "Halt stage"
                    font.pointSize: 10
                }
				Button {
	                id: button_halt
	                Layout.alignment: Qt.AlignCenter
	                text: "Halt"
                    font.pointSize: 10
	                onClicked: {
	                Julia.send_halt_stage()
	                if (button_tracking.text == "Stop") {
	                    button_tracking.text = Julia.toggle_tracking().toString()
	               		}
	            	}
	        	}
                Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "Recording"
                    font.pointSize: 10
                }
                Button {
                id: button_recording
                Layout.alignment: Qt.AlignCenter
                text: "Start"
                font.pointSize: 10
                onClicked: { button_recording.text = Julia.toggle_recording().toString() }
            	}
	        	Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "Crosshair"
                    font.pointSize: 10
                }
				Button {
	                id: button_crosshair
	                Layout.alignment: Qt.AlignCenter
	                text: "Show"
                    font.pointSize: 10
	                onClicked: { button_crosshair.text = Julia.toggle_crosshair().toString() }
	        	}
	        	Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "Ruler (100 um)"
                    font.pointSize: 10
                }
				Button {
	                id: button_ruler
	                Layout.alignment: Qt.AlignCenter
	                text: "Show"
                    font.pointSize: 10
	                onClicked: { button_ruler.text = Julia.toggle_ruler().toString() }
	        	}
                Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "Deepnet output"
                    font.pointSize: 10
                }
				Button {
	                id: button_deepnetoutput
	                Layout.alignment: Qt.AlignCenter
	                text: "Hide"
                    font.pointSize: 10
	                onClicked: { button_deepnetoutput.text = Julia.toggle_deepnetoutput().toString() }
	        	}
                Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "Set food coords "
                    font.pointSize: 10
                }

                Button {
	                id: button_TL
	                Layout.alignment: Qt.AlignCenter
	                text: "Set TL"
                    font.pointSize: 10
	                onClicked: { button_TL.text = Julia.set_TL_coord().toString() }
	        	}
                Button {
	                id: button_TR
	                Layout.alignment: Qt.AlignCenter
	                text: "Set TR"
                    font.pointSize: 10
	                onClicked: { button_TR.text = Julia.set_TR_coord().toString() }
	        	}
                Button {
	                id: button_BR
	                Layout.alignment: Qt.AlignCenter
	                text: "Set BR"
                    font.pointSize: 10
	                onClicked: { button_BR.text = Julia.set_BR_coord().toString() }
	        	}
                Button {
	                id: button_BL
	                Layout.alignment: Qt.AlignCenter
	                text: "Set BL"
                    font.pointSize: 10
	                onClicked: { button_BL.text = Julia.set_BL_coord().toString() }
	        	}
                

			}
        }
	}
    
    JuliaSignals {
        signal updateCanvas()
        onUpdateCanvas: img_canvas.update()
        
        signal updateTextSpeedAvg(var str_speed)
        onUpdateTextSpeedAvg: text_speed_avg.text = str_speed

        signal updateCoords(var x, var y)
        onUpdateCoords: {
                stage_x.text = x;
                stage_y.text = y;
            }
        
        signal updateFoodPatchCoords(var tl_x_, var tl_y_, var tr_x_, var tr_y_, var br_x_, var br_y_, var bl_x_, var bl_y_)
        onUpdateFoodPatchCoords: {
                tl_x.text = tl_x_;
                tl_y.text = tl_y_;
                tr_x.text = tr_x_;
                tr_y.text = tr_y_;
                br_x.text = br_x_;
                br_y.text = br_y_;
                bl_x.text = bl_x_;
                bl_y.text = bl_y_;
            }
        
        signal updateDistanceToPatch(var dist)
        onUpdateDistanceToPatch: {
                d2p.text = dist;
        }

        signal updateTextRecordingDuration(var str_recording_duration)
        onUpdateTextRecordingDuration: text_recording_duration.text = str_recording_duration
    }
    
    
}
