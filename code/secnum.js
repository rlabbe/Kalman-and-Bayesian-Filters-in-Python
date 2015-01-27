console.log("Section numbering...");

function number_sections(threshold) {

  var h1_number = 0;
  var h2_number = 0;

  if (threshold === undefined) {
    threshold = 2;  // does nothing so far
  }

  var cells = IPython.notebook.get_cells();
  
  for (var i=0; i < cells.length; i++) {

    var cell = cells[i];
    if (cell.cell_type !== 'heading') continue;
    
    var level = cell.level;
    if (level > threshold) continue;
    
    if (level === 1) {
        
        h1_number ++;
        var h1_element = cell.element.find('h1');
        var h1_html = h1_element.html();
        
        console.log("h1_html: " + h1_html);

        var patt = /^[0-9]+\.\s(.*)/;   // section number at start of string
        var title = h1_html.match(patt);  // just the title

        if (title != null) {  
          h1_element.html(h1_number + ". " + title[1]);
        }
        else {
          h1_element.html(h1_number + ". " + h1_html);
        }
        
        h2_number = 0;
        
    }
    
    if (level === 2) {
    
        h2_number ++;
        
        var h2_element = cell.element.find('h2');
        var h2_html = h2_element.html();

        console.log("h2_html: " + h2_html);

        
        var patt = /^[0-9]+\.[0-9]+\.\s/;
        var result = h2_html.match(patt);

        if (result != null) {
          h2_html = h2_html.replace(result, "");
        }

        h2_element.html(h1_number + "." + h2_number + ". " + h2_html);
        
    }
    
  }
  
}

number_sections();

// $([IPython.evnts]).on('create.Cell', number_sections);

$([IPython.events]).on('selected_cell_type_changed.Notebook', number_sections);

